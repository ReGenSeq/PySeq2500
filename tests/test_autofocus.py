"""Tests for the autofocus module."""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import xarray as xr
import dask.array as da

from pyseq2500.autofocus import Autofocus, autofocus, FocusFOV
from pyseq2500.utils import HW_CONFIG
from pre.image_analysis import HiSeqImages, sum_images
from xarray import DataArray


@pytest.fixture(scope="session")
def mock_microscope(dataarray_focus_stack):
    """Create a simple mock microscope.

    Session-scoped to allow reuse by session-scoped fixtures like
    mock_microscope_with_mocked_sum_image.
    """
    microscope = MagicMock()
    microscope.ZStage = MagicMock()
    microscope.ZStage.move = AsyncMock()
    microscope._move = AsyncMock()
    microscope._scan = AsyncMock()
    microscope._focus_stack = AsyncMock()
    microscope._focus_stack.return_value = dataarray_focus_stack
    microscope.Camera = MagicMock()
    microscope.Camera.config = HW_CONFIG["Cameras"]
    return microscope


@pytest_asyncio.fixture(scope="session")
def af(microscope, fc_A_roi):
    return Autofocus(microscope, fc_A_roi)


@pytest_asyncio.fixture(scope="session")
def af_with_rough_scan(af, rough_scan_hiseq):
    """Create Autofocus instance with real RoughScan data loaded."""

    af.rough_ims = rough_scan_hiseq.im
    return af


@pytest.fixture(scope="session")
def summed_rough_scan_image(rough_scan_hiseq):
    """Pre-compute summed image from RoughScan data once per session.

    This avoids repeatedly calling sum_images() which processes the full
    4452x6143 pixel images and is expensive (~5-8 seconds per call).

    Returns:
        tuple: (summed_image, median_value)
            - summed_image: numpy.ndarray - Summed image from all channels
            - median_value: float - Median value of summed image
    """
    summed = sum_images(rough_scan_hiseq.im)
    median = np.median(summed)

    return summed, median


@pytest.fixture
def mock_sum_images(monkeypatch, summed_rough_scan_image):
    """Mock sum_images to return pre-computed result.

    This avoids re-computing sum_images in every test that calls
    find_candidate_fovs(). The monkeypatch is function-scoped (as required
    by pytest), but the underlying data is session-scoped.
    """
    summed_image, median_value = summed_rough_scan_image

    def mock_sum_images_func(images):
        return summed_image

    monkeypatch.setattr("pyseq2500.autofocus.sum_images", mock_sum_images_func)

    return mock_sum_images_func


@pytest.fixture
def mock_median(monkeypatch, summed_rough_scan_image):
    """Mock np.median to return pre-computed median value.

    The monkeypatch is function-scoped (as required by pytest), but the
    underlying data is session-scoped.
    """
    _, median_value = summed_rough_scan_image

    def mock_median_func(arr, axis=None):
        return median_value

    # Patch where median is used (in autofocus module), not where it's defined
    monkeypatch.setattr("numpy.median", mock_median_func)

    return mock_median_func


@pytest.fixture
def mock_focus_stack(monkeypatch, dataarray_focus_stack):
    async def mock_focus_stack_func(z_init=None, z_last=None, n_frames=0, velocity=0):
        return dataarray_focus_stack

    monkeypatch.setattr(
        "pyseq2500.microscope.Microscope._focus_stack", mock_focus_stack_func
    )

    return mock_focus_stack_func


@pytest.fixture(scope="session")
def mock_microscope_with_mocked_sum_image(
    mock_microscope, fc_A_roi, rough_scan_hiseq, summed_rough_scan_image
):
    """Create Autofocus instance with pre-computed summed image for session.

    This fixture is session-scoped and uses the pre-computed summed image.
    However, since find_candidate_fovs() unconditionally calls sum_images(),
    tests that call this method should ALSO use mock_sum_images and mock_median
    fixtures to prevent the expensive computation.

    Returns:
        Autofocus: Configured Autofocus instance with sum_image and median set
    """
    summed_image, median_value = summed_rough_scan_image

    af = Autofocus(mock_microscope, fc_A_roi)
    af.rough_ims = rough_scan_hiseq.im
    af.sum_image = summed_image
    af.median = median_value

    return af


@pytest_asyncio.fixture
def af_with_mock_sum_image(af_with_rough_scan, summed_rough_scan_image):
    """Create Autofocus instance with pre-computed summed image for faster testing.

    Uses pre-computed summed image to avoid expensive sum_images computation
    in every test. Ideal for tests that call find_candidate_fovs multiple times.
    """
    summed_image, median_value = summed_rough_scan_image

    af_with_rough_scan.sum_image = summed_image
    af_with_rough_scan.median = median_value

    return af_with_rough_scan


@pytest_asyncio.fixture(scope="module")
def synthetic_focus_stack() -> DataArray:
    # Create a synthetic focus stack with object dtype containing image-like arrays
    n_frames = 50
    channels = [558, 610, 687, 740]
    n_row = 16
    n_col = 64

    n_channels = len(channels)
    focus_stack = np.empty((n_channels, n_frames), dtype=object)

    # Create Gaussian-like focus metric
    z_positions = np.arange(n_frames)
    focus_metric = np.exp(-((z_positions - 100) ** 2) / (2 * 20**2))

    # Each "image" has texture with contrast proportional to focus metric
    # Use a fixed pattern that varies spatially so Sobel detects edges
    np.random.seed(42)
    base_pattern = np.random.randint(0, 100, (n_row, n_col), dtype=np.uint16)

    for i in range(n_channels):
        for j in range(n_frames):
            # Scale pattern by focus metric to simulate focus variation
            focus_stack[i, j] = (base_pattern * focus_metric[j]).astype(np.uint16)

    xr_focus_stack = DataArray(
        focus_stack.tolist(),
        dims=["channel", "z", "row", "col"],
        coords={"channel": channels, "z": range(n_frames)},
    )

    return xr_focus_stack


@pytest_asyncio.fixture(scope="session")
def dataarray_focus_stack(focus_stack_data, fc_A_roi):
    # calculate Z steps
    n_frames = fc_A_roi.stage.n_frames
    velocity = 0.1
    spum = 235
    velocity = velocity * 1000 * spum  # step/s
    interval = 0.040202
    z = np.arange(n_frames) * interval * velocity  # step
    z_pos = (z + 2000).clip(max=62000)

    stack = []
    for i in range(2):
        n_cached_frames = len(focus_stack_data[i])
        if n_frames <= n_cached_frames:
            stack.append(focus_stack_data[i][0:n_frames])
        else:
            n_extra = n_frames - n_cached_frames
            last_frame = focus_stack_data[i][-1:]
            extra_frames = np.repeat(last_frame, n_extra, axis=0)
            stack.append(np.vstack([focus_stack_data[i], extra_frames]))

    hs_stack = HiSeqImages.open_objstack(np.hstack(stack), z=z_pos)
    hs_stack.correct_background()

    return hs_stack.im.compute()


@pytest.mark.asyncio
class TestAutofocus:
    """Test Autofocus class."""

    @pytest_asyncio.fixture(autouse=True)
    async def test_init(self, af):
        """Test Autofocus initialization."""
        await af.microscope._initialize()

        assert af.roi is not None

    async def test_find_best_z_empty_stack(self, af):
        """Test find_best_z with empty stack."""

        empty_stack = np.zeros((0, 0))
        result = af.find_best_z(empty_stack)

        assert result is None

    async def test_find_best_z_valid_stack(self, af, synthetic_focus_stack):
        """Test find_best_z with valid focus stack."""

        result = af.find_best_z(synthetic_focus_stack)

        assert result is not None
        assert isinstance(result, int)
        # Result should be a valid integer (actual value depends on ROI z_init offset)

    async def test_ransac_focus_insufficient_points(self, af):
        """Test RANSAC with insufficient points."""

        # Only 2 points - not enough for plane fitting
        focus_points = np.array([[1000, 2000, 5000], [1100, 2100, 5100]])

        result = af.ransac_focus(focus_points)

        assert result is None

    async def test_format_focus_data(self, af, synthetic_focus_stack):
        """Test focus data formatting."""

        n_channels, n_frames = synthetic_focus_stack.shape[0:2]
        result = af.format_focus_data(synthetic_focus_stack)

        assert result is not False
        assert result.shape == (n_frames, n_channels)
        # Result should be numeric dtype with focus metrics
        assert np.issubdtype(result.dtype, np.floating)

    async def test_format_focus_data_zero_signal(self, af):
        """Test focus data formatting with zero signal."""

        # Create focus stack with zero images
        n_frames = 100
        channels = [558, 610, 687, 740]

        n_channels = len(channels)
        focus_stack = np.empty((n_frames, n_channels), dtype=object)
        for i in range(n_frames):
            for j in range(n_channels):
                focus_stack[i, j] = np.zeros((16, 64), dtype=np.uint16)
        xr_focus_stack = DataArray(
            focus_stack.tolist(),
            dims=["z", "channel", "row", "col"],
            coords={"channel": channels, "z": range(n_frames)},
        )

        result = af.format_focus_data(xr_focus_stack)

        assert result is False

    async def test_capture_rough_scan(self, af):
        """Test rough scan capture."""

        result = await af.capture_rough_scan()
        assert result == af.roi.focus.output

    async def test_find_candidate_fovs_no_images(self, af):
        """Test FOV detection with no images."""
        af.rough_ims = None

        with pytest.raises(ValueError, match="No images available"):
            af.find_candidate_fovs()

    @pytest.mark.mock
    async def test_evaluate_fov(self, af, fc_A_roi):
        """Test FOV evaluation with real focus stack data from Zenodo."""

        # Only run for MockMicroscope (uses real Zenodo data)
        if af.microscope.name == "MockMicroscope":
            tile = TestFocusFOV.make_synthetic_fov(1000, 2000)
            fov = FocusFOV(1024, 16, tile, px_row=1500, px_col=20)
            fov = await af.evaluate_fov(fov, fc_A_roi)

            assert isinstance(fov, FocusFOV)
            assert isinstance(fov.x, int)
            assert isinstance(fov.y, int)
            assert isinstance(fov.z, int)


@pytest.mark.asyncio
class TestAutofocusFunction:
    """Test the main autofocus function."""

    async def test_autofocus_function(self, mock_microscope, fc_A_roi):
        """Test the autofocus convenience function."""
        # Mock the run method
        with patch.object(Autofocus, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = 5000

            result = await autofocus(mock_microscope, fc_A_roi)

            assert result == 5000
            mock_run.assert_called_once()


def make_synthetic_images(signal_low: int, signal_high: int):
    channels = [558, 610, 687, 740]
    x_tiles = [10292, 10607, 10922]
    # Create synthetic images for each channel and tile

    ch_stack = []
    for ch in channels:
        x_stack = []
        for x_pos in x_tiles:
            # Create a 4608x2048 synthetic image with varying patterns
            rows = 4608
            cols = 2048
            np.random.seed(ch * 1000 + x_pos)
            # Create background with some structure
            img = np.random.randint(100, 200, (rows, cols), dtype=np.int16)

            # Add some high-contrast features (simulating cells/FOVs)
            for _ in range(20):
                x = np.random.randint(0, rows)
                y = np.random.randint(0, cols)
                size = np.random.randint(10, 50)
                img[y : y + size, x : x + size] = np.random.randint(
                    signal_low, signal_high
                )

            x_stack.append(da.from_array(img))
        ch_stack.append(da.concatenate(x_stack, axis=1))

    xr_im = xr.DataArray(
        da.stack(ch_stack),
        name="RoughScan",
        dims=["channel", "row", "col"],
        coords={"channel": channels},
        attrs={"machine": "Test", "scale": 1},
    )

    return xr_im


class TestRoughScanFOVIdentification:
    """Comprehensive tests for FOV identification from rough scan images."""

    @pytest.fixture
    def af_with_mock(self, mock_microscope, fc_A_roi):
        """Create Autofocus instance with mock microscope."""
        return Autofocus(mock_microscope, fc_A_roi)

    def test_target_inliers(self, af):
        n_markers, roi = af.target_inliers()
        assert n_markers > 0
        assert roi == af.roi

    def test_load_images_from_nonexistent_directory(self, af):
        """Test loading images from non-existent directory raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            af.load_and_process_images(Path("/nonexistent/path"))

    def test_find_candidate_fovs_with_synthetic_images(self, af_with_mock):
        """Test finding candidate FOVs from synthetic rough scan images."""

        images = make_synthetic_images(1000, 2000)

        # Find candidate FOVs
        candidate_fovs = af_with_mock.find_candidate_fovs(images)

        # Verify results
        assert candidate_fovs is not None
        assert isinstance(candidate_fovs, list)
        assert len(candidate_fovs) > 0

        n_rows = len(images.row)
        n_cols = len(images.col)
        for fov in candidate_fovs:
            assert isinstance(fov.px_row, int)
            assert isinstance(fov.px_col, int)
            assert 0 < fov.px_row < n_rows
            assert 0 < fov.px_col < n_cols

    def test_find_candidate_fovs_with_low_signal(self, af_with_mock):
        """Test FOV detection with low signal images."""

        images = make_synthetic_images(200, 400)
        # Should handle low signal gracefully
        candidate_fovs = af_with_mock.find_candidate_fovs(images)

        # May return empty array or few candidates with low signal
        assert isinstance(candidate_fovs, list)
        assert len(candidate_fovs) > 0

    def test_find_candidate_fovs_with_uniform_images(self, af_with_mock):
        """Test FOV detection with uniform images (no features)."""

        images = make_synthetic_images(100, 200)
        candidate_fovs = af_with_mock.find_candidate_fovs(images)

        assert isinstance(candidate_fovs, list)
        assert len(candidate_fovs) == 0
        # Uniform images may result in empty or limited candidates

    def test_fov_ranking_with_different_n_candidates(self, af_with_mock):
        """Test FOV ranking with different numbers of requested candidates."""
        images = make_synthetic_images(1000, 2000)
        # Test with different candidate counts
        for n_candidates in [10, 25, 50]:
            candidate_fovs = af_with_mock.find_candidate_fovs(images, n_candidates)

            assert isinstance(candidate_fovs, list)
            assert len(candidate_fovs) == n_candidates
            # Should attempt to find at least the requested number (or available max)

    # def test_partial_focus_routine(self, af_with_simple_roi, synthetic_rough_scan_images):
    #     """Test FOV detection with 'partial once' focus routine."""
    #     af_with_simple_roi.focus_routine = "partial once"
    #     af_with_simple_roi.load_and_process_images(synthetic_rough_scan_images)

    #     candidate_fovs = af_with_simple_roi.find_candidate_fovs(n_candidates=20)
    #     assert isinstance(candidate_fovs, np.ndarray)

    # TODO
    # def test_candidate_fovs_deterministic_behavior(self, mock_af_with_synthetic_rough_scans):
    #     """Test that FOV detection produces consistent results."""

    #     # Run twice with same parameters
    #     fovs_1 = mock_af_with_synthetic_rough_scans.find_candidate_fovs(n_candidates=20)
    #     fovs_2 = mock_af_with_synthetic_rough_scans.find_candidate_fovs(n_candidates=20)

    #     # Should produce same number of candidates
    #     assert len(fovs_1) == len(fovs_2)

    #     # Should produce same coordinates (deterministic algorithm)
    #     if len(fovs_1) > 0:
    #         np.testing.assert_array_equal(fovs_1, fovs_2)

    def test_scale_parameter_effect(self, af_with_mock):
        """Test that scale parameter affects FOV detection."""
        images = make_synthetic_images(1000, 2000)
        # Test with different scale values
        af_with_mock.scale = 1
        fovs_scale_1 = af_with_mock.find_candidate_fovs(images)
        af_with_mock.scale = 2
        fovs_scale_2 = af_with_mock.find_candidate_fovs(images)

        # Both should return valid arrays
        assert isinstance(fovs_scale_1, list)
        assert isinstance(fovs_scale_2, list)
        assert len(fovs_scale_1) > 0
        assert len(fovs_scale_2) > 0


class TestFocusFOV:
    @staticmethod
    def make_synthetic_fov(
        signal_low: int, signal_high: int, nrows: int = 128, ncols: int = 128
    ) -> np.ndarray:
        np.random.seed(signal_low + signal_high)
        # Create background with some structure
        img = np.random.randint(550, 650, (nrows, ncols), dtype=np.int16)

        # Add some high-contrast features (simulating cells/FOVs)
        for _ in range(20):
            r = np.random.randint(0, nrows)
            c = np.random.randint(0, ncols)
            size = np.random.randint(10, 50)
            _r = max(nrows, r + size)
            _c = max(ncols, c + size)
            img[r:_r, c:_c] = np.random.randint(signal_low, signal_high)

        return img

    @pytest.fixture
    def synthetic_fov(self) -> FocusFOV:
        tile = TestFocusFOV.make_synthetic_fov(1000, 2000)
        return FocusFOV(1024, 16, tile)

    def test_focusfov_init(self):
        tile = TestFocusFOV.make_synthetic_fov(1000, 2000)
        fov = FocusFOV(0, 0, tile)
        assert fov.row_offset == 0
        assert fov.col_offset == 0
        assert (fov.tile == tile).all()

    def test_generate_mini_tiles(
        self, mock_microscope_with_mocked_sum_image, mock_sum_images, mock_median
    ):
        generated = False
        for fov in mock_microscope_with_mocked_sum_image.generate_mini_tiles():
            generated = True
            assert isinstance(fov, FocusFOV)

        assert generated

    def test_filter_empty_fov(
        self,
        mock_microscope_with_mocked_sum_image,
        synthetic_fov,
        mock_sum_images,
        mock_median,
    ):
        mock_microscope_with_mocked_sum_image.median = 600
        mock_microscope_with_mocked_sum_image.MAD = 40
        mock_microscope_with_mocked_sum_image.score_thresh = 40**2 / 600 * 3.5

        # Test FOV with high signal
        fov = mock_microscope_with_mocked_sum_image.filter_empty_fov(synthetic_fov)
        assert fov is not None
        assert fov.kurtosis >= 3.5
        assert fov.nv > 1

        # Test FOV with low signal
        tile_low = TestFocusFOV.make_synthetic_fov(750, 900)
        fov_low = FocusFOV(0, 0, tile_low)
        fov_low = mock_microscope_with_mocked_sum_image.filter_empty_fov(fov_low)
        assert fov_low is not None

        # Test FOV with no signal
        tile_neg = TestFocusFOV.make_synthetic_fov(550, 650)
        fov_neg = FocusFOV(0, 0, tile_neg)
        fov_neg = mock_microscope_with_mocked_sum_image.filter_empty_fov(fov_neg)
        assert fov_neg is None

    @pytest.mark.slow
    def test_find_candidate_fovs(
        self, mock_microscope_with_mocked_sum_image, mock_sum_images, mock_median
    ):
        fovs = mock_microscope_with_mocked_sum_image.find_candidate_fovs()

        assert len(fovs) > 0
        assert fovs[0].score > fovs[-1].score


class TestRoughScanRealDataFOVIdentification:
    """Tests for FOV identification using real RoughScan data from Zenodo.

    These tests verify the complete workflow of identifying FOVs after
    an out-of-focus rough scan using real test data downloaded via pooch.
    """

    def test_find_candidate_fovs_with_real_data(
        self, mock_microscope_with_mocked_sum_image, mock_sum_images, mock_median
    ):
        """Test finding candidate FOVs from real RoughScan data."""
        # Find candidate FOVs

        af = mock_microscope_with_mocked_sum_image
        candidates = af.find_candidate_fovs(n_candidates=50)

        # Verify results
        assert candidates is not None
        assert isinstance(candidates, list)
        assert len(candidates) > 0, "Should find candidate FOVs from real data"

        n_rows = len(af.rough_ims.row)
        n_cols = len(af.rough_ims.col)

        for fov in candidates:
            assert isinstance(fov.px_row, int)
            assert isinstance(fov.px_col, int)
            assert 0 < fov.px_row < n_rows
            assert 0 < fov.px_col < n_cols

        # TODO "All FOV candidates should be unique (no duplicates)"
        # unique_candidates = np.unique(candidates, axis=0)
        # assert len(unique_candidates) == len(candidates)

    def test_find_candidate_fovs_different_counts(
        self, mock_microscope_with_mocked_sum_image, mock_sum_images, mock_median
    ):
        """Test FOV detection with different candidate counts."""

        for n_candidates in [10, 20]:
            candidates = mock_microscope_with_mocked_sum_image.find_candidate_fovs(
                n_candidates=n_candidates
            )

            assert isinstance(candidates, list)
            assert len(candidates) == n_candidates

    # def test_find_candidate_fovs_partial_routine(self, mock_microscope, simple_roi, rough_scan_hiseq):
    #     """Test FOV detection with 'partial once' focus routine."""
    #     simple_roi.focus.routine = "partial once"
    #     af = Autofocus(mock_microscope, simple_roi)
    #     af.rough_ims = rough_scan_hiseq

    #     candidates = af.find_candidate_fovs(n_candidates=50)

    #     assert isinstance(candidates, np.ndarray)
    #     assert len(candidates) > 0

    # def test_find_candidate_fovs_full_routine(self, mock_microscope, simple_roi, rough_scan_hiseq):
    #     """Test FOV detection with 'full once' focus routine."""
    #     simple_roi.focus.routine = "full once"
    #     af = Autofocus(mock_microscope, simple_roi)
    #     af.rough_ims = rough_scan_hiseq

    #     candidates = af.find_candidate_fovs(n_candidates=50)

    #     assert isinstance(candidates, np.ndarray)
    #     assert len(candidates) > 0

    # TODO
    # def test_find_candidate_fovs_deterministic(self, af_with_rough_scan):
    #     """Test that FOV detection is deterministic."""

    #     candidates_1 = af_with_rough_scan.find_candidate_fovs(n_candidates=50)
    #     candidates_2 = af_with_rough_scan.find_candidate_fovs(n_candidates=50)

    #     # Should produce identical results
    #     np.testing.assert_array_equal(candidates_1, candidates_2)

    # def test_find_candidate_fovs_spatial_distribution(self, af_with_rough_scan):
    #     """Test that FOVs have reasonable spatial distribution."""
    #     af = af_with_rough_scan

    #     candidates = af.find_candidate_fovs(n_candidates=50)

    #     if len(candidates) > 1:
    #         # Calculate spatial spread
    #         row_spread = np.max(candidates[:, 0]) - np.min(candidates[:, 0])
    #         col_spread = np.max(candidates[:, 1]) - np.min(candidates[:, 1])

    #         # FOVs should have some spatial spread
    #         # (may be concentrated in one dimension depending on algorithm)
    #         assert row_spread > 0 or col_spread > 0, \
    #             "FOVs should not all be at the exact same location"

    def test_scale_parameter_affects_fov_count(
        self, mock_microscope_with_mocked_sum_image, mock_sum_images, mock_median
    ):
        """Test that scale parameter affects number of FOVs found."""

        # Test with different scale values
        mock_microscope_with_mocked_sum_image.scale = 1
        fovs_scale_1 = mock_microscope_with_mocked_sum_image.find_candidate_fovs(
            n_candidates=50
        )

        mock_microscope_with_mocked_sum_image.scale = 2
        fovs_scale_2 = mock_microscope_with_mocked_sum_image.find_candidate_fovs(
            n_candidates=50
        )

        # Both should return valid arrays
        assert isinstance(fovs_scale_1, list)
        assert isinstance(fovs_scale_2, list)
        assert len(fovs_scale_1) > 0
        assert len(fovs_scale_2) > 0

    def test_ransac(self, af_with_mock_sum_image, mock_sum_images, mock_median):
        af = af_with_mock_sum_image
        fovs = af.find_candidate_fovs(n_candidates=50)

        np.random.seed(8387)
        n_points = 6
        points = []
        for i in range(n_points):
            x_step, y_step = af.microscope.px_to_step(
                fovs[i].px_row, fovs[i].px_col, af.roi, af.scale
            )
            points.append([x_step, y_step])
        points = np.array(points)

        # Plane: z = 0.01*x + 0.001*y + 5000
        std = af.microscope.ZStage.config["spum"] * 5
        z = (
            0.001 * points[:, 0]
            + 0.005 * points[:, 1]
            + 20000
            + np.random.normal(0, std / 2, n_points)
        )

        df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "z": z})

        model = af.ransac_focus(df, n_points)

        assert model is not None
        _z = model.predict(points)

        assert (np.abs(_z - z) < std).all()
        _z = model.predict(df.loc[[0], ["x", "y"]])
        _z = round(_z[0])
        assert isinstance(_z, int)
        assert abs(_z - z[0]) < std


class TestFOVMapVisualization:
    """Tests for the FOV map visualization functionality."""

    @pytest.fixture
    def af_with_evaluated_fovs(self, af_with_rough_scan):
        """Create Autofocus instance with some evaluated FOVs."""

        fovs = [
            FocusFOV(
                0, 0, np.array(0), px_row=100, px_col=200, z=5000, inlier=True, rank=0
            ),
            FocusFOV(
                0, 0, np.array(0), px_row=150, px_col=250, z=5100, inlier=True, rank=1
            ),
            FocusFOV(
                0, 0, np.array(0), px_row=200, px_col=300, z=4900, inlier=False, rank=2
            ),
            FocusFOV(
                0, 0, np.array(0), px_row=250, px_col=350, z=5050, inlier=True, rank=3
            ),
            FocusFOV(
                0, 0, np.array(0), px_row=120, px_col=300, z=5000, inlier=True, rank=4
            ),
        ]

        af_with_rough_scan.focus_df = pd.DataFrame(fovs)
        af_with_rough_scan.fov_shape = (16, 64)  # Example FOV dimensions
        af_with_rough_scan.optimal_z = 5000

        return af_with_rough_scan

    @pytest.fixture
    def candidate_fovs(self):
        """Create Autofocus instance with some evaluated FOVs."""

        fovs = [
            FocusFOV(0, 0, np.array(0), px_row=100, px_col=200, z=5000, rank=0),
            FocusFOV(0, 0, np.array(0), px_row=150, px_col=250, z=5100, rank=1),
            FocusFOV(0, 0, np.array(0), px_row=200, px_col=300, z=4900, rank=2),
            FocusFOV(0, 0, np.array(0), px_row=250, px_col=350, z=5050, rank=3),
            FocusFOV(0, 0, np.array(0), px_row=120, px_col=300, z=5000, rank=4),
            FocusFOV(0, 0, np.array(0), px_row=101, px_col=301, rank=5),
            FocusFOV(0, 0, np.array(0), px_row=102, px_col=302, rank=6),
            FocusFOV(0, 0, np.array(0), px_row=103, px_col=303, rank=7),
            FocusFOV(0, 0, np.array(0), px_row=104, px_col=304, rank=8),
            FocusFOV(0, 0, np.array(0), px_row=105, px_col=305, rank=9),
        ]

        return fovs

    def test_map_and_save_focus_fovs_basic(
        self, af_with_evaluated_fovs, candidate_fovs, tmp_path
    ):
        """Test basic FOV map generation and saving."""
        af = af_with_evaluated_fovs

        output_path = tmp_path / "test_fov_map.png"
        result_path = af.map_and_save_focus_fovs(candidate_fovs, output_path)

        # Verify file was created
        assert result_path.exists()
        assert result_path == output_path
        assert result_path.suffix == ".png"

    def test_map_and_save_focus_fovs_default_path(
        self, af_with_evaluated_fovs, candidate_fovs, test_directory
    ):
        """Test FOV map saves to default path when not specified."""
        af = af_with_evaluated_fovs
        af_with_evaluated_fovs.focus_output = test_directory / "focus"

        result_path = af.map_and_save_focus_fovs(candidate_fovs)

        # Should save to default location
        assert result_path.exists()
        assert "FOV_Map_" in result_path.name

    def test_map_and_save_focus_fovs_unevaluated_only(
        self, af_with_evaluated_fovs, candidate_fovs, tmp_path
    ):
        """Test FOV map with only unevaluated candidates."""

        # Clear evaluated dataframe
        af = af_with_evaluated_fovs
        af.focus_df = af.focus_df.iloc[0:0]

        # select unevaluated fovs
        _fovs = candidate_fovs[5:]
        _ = []
        for i, f in enumerate(_fovs):
            f.rank = i
            _.append(f)

        output_path = tmp_path / "unevaluated_map.png"
        result_path = af.map_and_save_focus_fovs(_, output_path)

        assert result_path.exists()

    def test_map_and_save_focus_fovs_no_images_raises_error(
        self, microscope, fc_A_roi, candidate_fovs
    ):
        """Test that FOV map raises error when no images loaded."""

        af = Autofocus(microscope, fc_A_roi)

        with pytest.raises(ValueError, match="No rough scan images"):
            af.map_and_save_focus_fovs(candidate_fovs)


@pytest.mark.slow
@pytest.mark.mock
@pytest.mark.asyncio
async def test_autofocus_integration(
    microscope, fc_A_roi, rough_scan_data, dataarray_focus_stack
):
    """Integration test: Partial workflow from RoughScan to FOV identification.

    Tests the complete FOV identification pipeline:
    1. Load RoughScan images from real Zenodo data
    2. Find candidate FOVs
    3. Evaluate FOVs and run RANSAC
    4. Generate and save FOV map visualization
    """

    # Initialize microscope instruments
    await microscope._initialize()

    focus_dir = fc_A_roi.focus.output

    # Set focus output to rough_scan_data so images are loaded from there
    fc_A_roi.focus.output = rough_scan_data

    af = Autofocus(microscope, fc_A_roi)
    # Override focus_output to save FOV map to temp directory
    af.focus_output = focus_dir

    opt_z = await af.run()
    assert 26000 < opt_z < 28000

    # Verify FOV map PNG was saved to temp focus directory
    result_path = focus_dir / f"FOV_Map_{fc_A_roi.name}.png"
    assert result_path.exists(), f"FOV map should be saved to {result_path}"
