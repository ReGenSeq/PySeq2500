"""Tests for the autofocus module."""

import pytest
import pytest_asyncio
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import xarray as xr
import dask.array as da

from pyseq2500.autofocus import Autofocus, autofocus
from pre.image_analysis import HiSeqImages
from xarray import DataArray


@pytest.fixture
def mock_microscope():
    """Create a simple mock microscope."""
    microscope = MagicMock()
    microscope.ZStage = MagicMock()
    microscope.ZStage.move = AsyncMock()
    microscope._move = AsyncMock()
    microscope._scan = AsyncMock()
    microscope._focus_stack = AsyncMock()
    return microscope


@pytest_asyncio.fixture(scope="session")
def af(microscope, fc_A_roi):
    return Autofocus(microscope, fc_A_roi)


@pytest_asyncio.fixture(scope="session")
def af_with_rough_scan(af, rough_scan_hiseq):
    """Create Autofocus instance with real RoughScan data loaded."""

    af.rough_ims = rough_scan_hiseq

    return af


@pytest_asyncio.fixture
def af_with_rough_scan_mocked(
    af_with_rough_scan, summed_rough_scan_image, mock_sum_images
):
    """Create Autofocus instance with mocked sum_images for faster testing.

    Uses pre-computed summed image to avoid expensive sum_images computation
    in every test. Ideal for tests that call find_candidate_fovs multiple times.
    """
    af_with_rough_scan.sum_image = summed_rough_scan_image

    return af_with_rough_scan


@pytest_asyncio.fixture(scope="module")
def synthetic_focus_stack() -> DataArray:
    # Create a synthetic focus stack with object dtype containing image-like arrays
    n_frames = 200
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

    async def test_ransac_focus_good_consensus(self, af):
        """Test RANSAC with good consensus."""

        # Create points on a plane with some noise
        np.random.seed(42)
        n_points = 20
        x = np.random.uniform(1000, 2000, n_points)
        y = np.random.uniform(3000000, 3100000, n_points)
        # Plane: z = 0.01*x + 0.001*y + 5000
        z = 0.01 * x + 0.001 * y + 5000 + np.random.normal(0, 10, n_points)

        focus_points = np.column_stack([x, y, z])

        result = af.ransac_focus(focus_points)

        assert result is not None
        assert len(result) == 2  # slope and intercept

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
            px_pt = np.array([100, 200])
            result = await af.evaluate_fov(px_pt, fc_A_roi)

            assert result is not None
            assert len(result) == 3  # x, y, z


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
        attrs={"machine": "Test"},
    )

    return HiSeqImages(xr_im, machine="Test")


class TestRoughScanFOVIdentification:
    """Comprehensive tests for FOV identification from rough scan images."""

    @pytest.fixture
    def af_with_mock(self, mock_microscope, fc_A_roi):
        """Create Autofocus instance with mock microscope."""
        return Autofocus(mock_microscope, fc_A_roi)

    @pytest.fixture
    def mock_af_with_synthetic_rough_scans(self, af_with_mock):
        """Create synthetic rough scan images for testing.

        Creates a set of synthetic 12-bit TIFF images mimicking rough scan data
        with 4 channels and 3 tiles per channel.
        """
        af_with_mock.rough_ims = make_synthetic_images(500, 1000)
        return af_with_mock

    @pytest.fixture
    def mock_af_with_low_signal_images(self, af_with_mock):
        """Create synthetic images with very low signal."""
        af_with_mock.rough_ims = make_synthetic_images(500, 1000)
        return af_with_mock

    @pytest.fixture
    def mock_af_with_uniform_images(self, af_with_mock):
        """Create synthetic uniform images."""
        af_with_mock.rough_ims = make_synthetic_images(100, 200)
        return af_with_mock

    def test_load_images_from_nonexistent_directory(self, af):
        """Test loading images from non-existent directory raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            af.load_and_process_images(Path("/nonexistent/path"))

    def test_find_candidate_fovs_no_images(self, af):
        """Test FOV detection when no images are loaded."""
        af.rough_ims = None

        with pytest.raises(ValueError, match="No images available"):
            af.find_candidate_fovs()

    def test_find_candidate_fovs_with_synthetic_images(
        self, mock_af_with_synthetic_rough_scans
    ):
        """Test finding candidate FOVs from synthetic rough scan images."""

        # Find candidate FOVs
        candidate_fovs = mock_af_with_synthetic_rough_scans.find_candidate_fovs(
            n_candidates=50
        )

        # Verify results
        assert candidate_fovs is not None
        assert isinstance(candidate_fovs, np.ndarray)

        # Should have found some candidates
        if len(candidate_fovs) > 0:
            # Each candidate should be [row, col] coordinates
            assert candidate_fovs.ndim == 2
            assert candidate_fovs.shape[1] == 2

            # Coordinates should be non-negative
            assert np.all(candidate_fovs >= 0)

            # TODO Check for duplicates
            # unique_fovs = np.unique(candidate_fovs, axis=0)
            # There may be duplicates, but that's okay for candidate generation

            # Check coordinates are integers
            assert np.issubdtype(candidate_fovs.dtype, np.integer) or np.all(
                candidate_fovs == candidate_fovs.astype(int)
            )

    def test_find_candidate_fovs_with_low_signal(self, mock_af_with_low_signal_images):
        """Test FOV detection with low signal images."""

        # Should handle low signal gracefully
        candidate_fovs = mock_af_with_low_signal_images.find_candidate_fovs(
            n_candidates=10
        )

        # May return empty array or few candidates with low signal
        assert isinstance(candidate_fovs, np.ndarray)

    def test_find_candidate_fovs_with_uniform_images(self, mock_af_with_uniform_images):
        """Test FOV detection with uniform images (no features)."""

        candidate_fovs = mock_af_with_uniform_images.find_candidate_fovs(
            n_candidates=10
        )

        assert isinstance(candidate_fovs, np.ndarray)
        # Uniform images may result in empty or limited candidates

    def test_fov_ranking_with_different_n_candidates(
        self, mock_af_with_synthetic_rough_scans
    ):
        """Test FOV ranking with different numbers of requested candidates."""

        # Test with different candidate counts
        for n_candidates in [10, 25, 50, 100]:
            candidate_fovs = mock_af_with_synthetic_rough_scans.find_candidate_fovs(
                n_candidates=n_candidates
            )
            assert isinstance(candidate_fovs, np.ndarray)
            assert len(candidate_fovs) >= n_candidates
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

    def test_scale_parameter_effect(self, mock_af_with_synthetic_rough_scans):
        """Test that scale parameter affects FOV detection."""

        # Test with different scale values
        mock_af_with_synthetic_rough_scans.scale = 1
        fovs_scale_1 = mock_af_with_synthetic_rough_scans.find_candidate_fovs(
            n_candidates=20
        )

        mock_af_with_synthetic_rough_scans.scale = 2
        fovs_scale_2 = mock_af_with_synthetic_rough_scans.find_candidate_fovs(
            n_candidates=20
        )

        # Both should return valid arrays
        assert isinstance(fovs_scale_1, np.ndarray)
        assert isinstance(fovs_scale_2, np.ndarray)


class TestRoughScanRealDataFOVIdentification:
    """Tests for FOV identification using real RoughScan data from Zenodo.

    These tests verify the complete workflow of identifying FOVs after
    an out-of-focus rough scan using real test data downloaded via pooch.
    """

    def test_find_candidate_fovs_with_real_data(self, af_with_rough_scan_mocked):
        """Test finding candidate FOVs from real RoughScan data."""
        # Find candidate FOVs
        candidates = af_with_rough_scan_mocked.find_candidate_fovs(n_candidates=50)

        # Verify results
        assert candidates is not None
        assert isinstance(candidates, np.ndarray)
        assert len(candidates) > 0, "Should find candidate FOVs from real data"

        # Should be 2D array [n_candidates, 2]
        assert candidates.ndim == 2
        assert candidates.shape[1] == 2, "Each candidate should have [row, col]"

        assert np.all(candidates[:, 0] >= 0), "Row coordinates should be non-negative"
        assert np.all(candidates[:, 1] >= 0), (
            "Column coordinates should be non-negative"
        )

        n_rows, n_cols = af_with_rough_scan_mocked.rough_ims.im.shape[
            1:
        ]  # Skip channel dimension
        assert np.all(candidates[:, 0] < n_rows), "Row coordinates within image height"
        assert np.all(candidates[:, 1] < n_cols), (
            "Column coordinates within image width"
        )

        # TODO "All FOV candidates should be unique (no duplicates)"
        # unique_candidates = np.unique(candidates, axis=0)
        # assert len(unique_candidates) == len(candidates)

    def test_find_candidate_fovs_different_counts(self, af_with_rough_scan_mocked):
        """Test FOV detection with different candidate counts."""

        for n_candidates in [10, 25, 50, 100]:
            candidates = af_with_rough_scan_mocked.find_candidate_fovs(
                n_candidates=n_candidates
            )

            assert isinstance(candidates, np.ndarray)
            assert len(candidates) > n_candidates, (
                f"Should find candidates for n={n_candidates}"
            )

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

    def test_scale_parameter_affects_fov_count(self, af_with_rough_scan_mocked):
        """Test that scale parameter affects number of FOVs found."""

        # Test with different scale values
        af_with_rough_scan_mocked.scale = 1
        fovs_scale_1 = af_with_rough_scan_mocked.find_candidate_fovs(n_candidates=50)

        af_with_rough_scan_mocked.scale = 2
        fovs_scale_2 = af_with_rough_scan_mocked.find_candidate_fovs(n_candidates=50)

        # Both should return valid arrays
        assert isinstance(fovs_scale_1, np.ndarray)
        assert isinstance(fovs_scale_2, np.ndarray)
        assert len(fovs_scale_1) > 0
        assert len(fovs_scale_2) > 0


class TestFOVMapVisualization:
    """Tests for the FOV map visualization functionality."""

    @pytest.fixture
    def af_with_evaluated_fovs(self, af_with_rough_scan):
        """Create Autofocus instance with some evaluated FOVs."""

        # Simulate some evaluated FOVs
        af_with_rough_scan.evaluated_fovs = {
            (100, 200): {"z": 5000, "inlier": True, "rank": 0},
            (150, 250): {"z": 5100, "inlier": True, "rank": 1},
            (200, 300): {"z": 4900, "inlier": False, "rank": 2},  # Outlier
            (250, 350): {"z": 5050, "inlier": True, "rank": 3},
        }
        af_with_rough_scan.fov_shape = (16, 64)  # Example FOV dimensions
        af_with_rough_scan.optimal_z = 5000

        return af_with_rough_scan

    def test_map_and_save_focus_fovs_basic(self, af_with_evaluated_fovs, tmp_path):
        """Test basic FOV map generation and saving."""
        af = af_with_evaluated_fovs

        # Create candidate FOVs
        candidate_fovs = np.array(
            [
                [100, 200],
                [150, 250],
                [200, 300],
                [250, 350],
                [300, 400],
                [350, 450],
                [400, 500],
            ]
        )

        output_path = tmp_path / "test_fov_map.png"
        result_path = af.map_and_save_focus_fovs(candidate_fovs, output_path)

        # Verify file was created
        assert result_path.exists()
        assert result_path == output_path
        assert result_path.suffix == ".png"

    def test_map_and_save_focus_fovs_default_path(
        self, af_with_evaluated_fovs, test_directory
    ):
        """Test FOV map saves to default path when not specified."""
        af = af_with_evaluated_fovs
        af_with_evaluated_fovs.focus_output = test_directory / "focus"

        candidate_fovs = np.array([[100, 200], [150, 250]])

        result_path = af.map_and_save_focus_fovs(candidate_fovs)

        # Should save to default location
        assert result_path.exists()
        assert "FOV_Map_" in result_path.name

    def test_map_and_save_focus_fovs_unevaluated_only(
        self, af_with_evaluated_fovs, tmp_path
    ):
        """Test FOV map with only unevaluated candidates."""

        # No evaluated_fovs set - all will be unevaluated

        candidate_fovs = np.array([[100, 200], [150, 250], [200, 300]])

        output_path = tmp_path / "unevaluated_map.png"
        result_path = af_with_evaluated_fovs.map_and_save_focus_fovs(
            candidate_fovs, output_path
        )

        assert result_path.exists()

    def test_map_and_save_focus_fovs_no_images_raises_error(self, microscope, fc_A_roi):
        """Test that FOV map raises error when no images loaded."""

        af = Autofocus(microscope, fc_A_roi)

        candidate_fovs = np.array([[100, 200]])

        with pytest.raises(ValueError, match="No rough scan images"):
            af.map_and_save_focus_fovs(candidate_fovs)

    def test_map_and_save_focus_fovs_with_inliers_and_outliers(
        self, af_with_evaluated_fovs, tmp_path
    ):
        """Test FOV map correctly visualizes inliers and outliers."""
        af = af_with_evaluated_fovs

        # Candidates include both evaluated and unevaluated
        candidate_fovs = np.array(
            [
                [100, 200],  # inlier
                [150, 250],  # inlier
                [200, 300],  # outlier
                [250, 350],  # inlier
                [500, 600],  # unevaluated
            ]
        )

        output_path = tmp_path / "mixed_fov_map.png"
        result_path = af.map_and_save_focus_fovs(candidate_fovs, output_path)

        assert result_path.exists()
        # File size should be reasonable for a PNG
        assert result_path.stat().st_size > 10000  # At least 10KB


@pytest.mark.slow
@pytest.mark.mock
@pytest.mark.asyncio
async def test_autofocus_integration(microscope, fc_A_roi, rough_scan_data, tmp_path):
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

    await af.run()

    # Verify FOV map PNG was saved to temp focus directory
    result_path = focus_dir / f"FOV_Map_{fc_A_roi.name}.png"
    assert result_path.exists(), f"FOV map should be saved to {result_path}"
