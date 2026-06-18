"""Autofocus module for PySeq2500.

This module provides autofocus functionality using the Microscope class.
The autofocus routine:
1. Takes an out-of-focus scan using Microscope._scan
2. Opens images with pyseq_image.pre
3. Finds and ranks promising FOVs to focus on
4. Iteratively evaluates FOVs using focus stacks and RANSAC for consensus
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Union, Generator, List
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import sobel as ndimage_sobel
from scipy.stats import median_abs_deviation
import logging
from operator import attrgetter
from datetime import datetime

from pre.image_analysis import (
    HiSeqImages,
    sum_images,
)
from xarray import DataArray
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

# from attrs import define, field
from dataclasses import dataclass, fields
from skimage.filters import sobel_v

from pyseq_core.base_protocol import BaseROI
from pyseq_core.base_system import BaseMicroscope

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.set_loglevel("info")
matplotlib.use("Agg")  # Non-interactive backend


LOGGER = logging.getLogger("PySeq")


@dataclass
class FocusFOV:
    row_offset: int
    col_offset: int
    tile: Union[DataArray, np.ndarray]
    px_row: Optional[int] = None
    px_col: Optional[int] = None
    kurtosis: Optional[float] = None
    nv: Optional[float] = None  # normalized variance
    score: Optional[float] = None  # kurtosis * nv
    x: Optional[Union[float, int]] = None  # x stage step
    y: Optional[Union[float, int]] = None  # y stage step
    z: Optional[Union[float, int]] = None  # best z
    rank: Optional[int] = None
    inlier: bool = False  # for RANSAC plan fitting


def get_empty_focus_df() -> pd.DataFrame:
    # Create a dictionary of column names and types
    type_mapping = {f.name: f.type for f in fields(FocusFOV)}
    # Initialize and cast (Note: Python types like 'str' map to 'object' in Pandas)
    df = pd.DataFrame(columns=type_mapping.keys())
    df = df.astype({k: v for k, v in type_mapping.items() if v in [int, float, bool]})
    return df


class Autofocus:
    """Autofocus using Microscope class.

    The autofocus will follow the routine set in the ROI focus parameters.

    Attributes:
        microscope: Microscope instance for hardware control
        roi: Region of interest with focus parameters
        rough_ims: HiSeqImages object containing out-of-focus images
        scale: Down scale factor for thumbnails
        focus_points: Array of found focus positions [x, y, z, index]
    """

    def __init__(self, microscope: BaseMicroscope, roi: BaseROI):
        """Initialize Autofocus.

        Args:
            microscope: Microscope instance
            roi: ROI with focus parameters (optics, routine, output)
        """
        self.microscope: BaseMicroscope = microscope
        self.roi: BaseROI = roi
        self.rough_ims: Optional[DataArray] = None
        self.sum_image: Optional[DataArray] = None
        self.median: Union[float, int] = 1  # median of sum_image
        self.MAD: Union[float, int] = 1  # mean absolute deviation of sum_image
        self.score_thresh: float = 1.0  # used to filter empty FOVs, 3.5*MAD**2/median
        self.scale = 1  # Default scale, can be adjusted based on config
        self.focus_points = None

        # State tracking for FOV mapping
        # self.evaluated_fovs = {}  # {(row, col): {'z': z_val, 'inlier': bool, 'rank': int}}
        self.focus_df: pd.DataFrame = get_empty_focus_df()
        self.fov_shape = None  # (height, width) from focus stack images
        self.optimal_z = None  # Final optimal Z position
        # self.inlier_mask = None  # Boolean mask for inlier FOVs

        # Focus parameters from ROI
        self.focus_output = roi.focus.output if roi.focus.output else Path(".")
        self.focus_routine = roi.focus.routine if roi.focus.routine else "full once"

    async def capture_rough_scan(self, roi: Optional[BaseROI] = None):
        """Capture out-of-focus scan for finding candidate FOVs.

        Args:
            roi: Optional ROI override, uses self.roi if not provided

        Returns:
            Path to image directory containing rough scan images
        """
        if roi is None:
            roi = self.roi

        LOGGER.info("Autofocus:: Capturing rough scan for focus analysis")

        # Make roi specific image directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.focus_output = Path(roi.focus.output) / f"{roi.name}_{timestamp}"
        self.focus_output.mkdir(parents=True, exist_ok=True)

        # Move to ROI
        await self.microscope._move(
            x=roi.stage.x_init,
            y=roi.stage.y_init,
            z=roi.stage.z_init,
            tilt1=roi.stage.tilt1,
            tilt2=roi.stage.tilt2,
            tilt3=roi.stage.tilt3,
        )

        # Set laser power, laser filters, and exposure
        await self.microscope._set_parameters(roi.focus.optics)

        # Set Z position slightly out of focus for rough scan
        z_last = roi.stage.z_init + 1

        # Capture rough scan
        await self.microscope._scan(
            roi=roi,
            name=f"RoughScan{roi.name}",
            image_dir=self.focus_output,
            z_last=z_last,
        )

        return self.focus_output

    def load_and_process_images(self, image_dir: Union[str, Path, None]):
        """Load and process rough scan images.

        Args:
            image_dir: Directory containing rough scan images

        Returns:
            HiSeqImages object with loaded
        """

        if image_dir is None:
            image_dir = self.focus_output

        LOGGER.info(f"Autofocus:: Loading images from {image_dir}")

        # Load images using pyseq_image.pre
        rough_ims = HiSeqImages.open_RoughScan(str(image_dir), ["tif", "tiff"])

        # Correct background if method exists
        if hasattr(self.rough_ims, "correct_background"):
            rough_ims.correct_background()
        # Correct background if method exists
        if hasattr(self.rough_ims, "register_channels"):
            rough_ims.register_channels()

        if hasattr(rough_ims, "im"):
            self.rough_ims = rough_ims.im
        else:
            self.rough_ims = rough_ims

        LOGGER.debug(self.rough_ims)

        return self.rough_ims

    def generate_mini_tiles(
        self, width=128, height=128
    ) -> Generator[FocusFOV, None, None]:
        scale = self.sum_image.scale

        edge_c = self.microscope.Camera.config.get("sensor_width", 2048)
        edge_r = self.microscope.Camera.config.get("sensor_height", edge_c)

        edge_c = edge_c // 2 // scale
        edge_r = edge_r // 2 // scale

        n_rows, n_cols = self.sum_image.shape

        if hasattr(self.sum_image, "values"):
            self.sum_image = self.sum_image.values

        for r in range(edge_r, n_rows - edge_r, height):
            _r = min(r + height, n_rows - edge_r)
            for c in range(edge_c, n_cols - edge_c, width):
                _c = min(c + width, n_cols - edge_c)
                tile = self.sum_image[r:_r, c:_c]
                yield FocusFOV(r, c, tile)

    def filter_empty_fov(
        self, fov: FocusFOV, kurtosis_thresh=3.5, nv_thresh=1.0
    ) -> Union[FocusFOV, None]:
        def median(arr, axis):
            return self.median

        # Kurtosis based on global median
        if hasattr(fov.tile, "values"):
            fov.tile = fov.tile.values
        MAD = median_abs_deviation(fov.tile, center=median, axis=None)
        fov.kurtosis = np.mean(
            (((fov.tile - self.median) / (self.MAD * 1.4826)) ** 4), axis=None
        )
        # Normalized variance: nv
        fov.nv = MAD**2 / self.median

        # LOGGER.debug(f"k={fov.kurtosis}, nv={fov.nv}")

        fov.score = fov.kurtosis * fov.nv
        if fov.score >= self.score_thresh:
            return fov
        else:
            return None

    @staticmethod
    def refine_fov(
        fov: FocusFOV, px_percentile: Union[int, float] = 95, **kwargs
    ) -> FocusFOV:
        threshold = np.percentile(fov.tile, px_percentile)

        # 2. Get coordinates of pixels above threshold
        coords = np.argwhere(fov.tile >= threshold)

        # 3. Calculate the mean position (Centroid)
        fov.px_row = int(np.mean(coords[:, 0])) + fov.row_offset
        fov.px_col = int(np.mean(coords[:, 1])) + fov.col_offset

        # Need baseroi and scale
        # fov.x_step, fov.y_step = self.microscope.px_to_step(fov.px_row, fov.px_col)

        return fov

    @staticmethod
    def spatial_uniform_sort(fovs: List[FocusFOV]) -> List[FocusFOV]:
        num_points = len(fovs)
        if num_points == 0:
            return []
        else:
            points = np.array([[fov.px_row, fov.px_col] for fov in fovs])

        # Initialize
        sorted_indices = [0]
        # Track the minimum distance of every point to the current set
        min_distances = np.linalg.norm(points - points[0], axis=1)

        fov = fovs[0]
        fov.rank = 0
        sorted_fovs = [fov]
        for _ in range(1, num_points):
            # Pick the point that is furthest from its nearest neighbor in the set
            next_index = np.argmax(min_distances)
            sorted_indices.append(next_index)

            # Update min_distances: it's the min of the old distance
            # and the distance to the new point
            new_distances = np.linalg.norm(points - points[next_index], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

            fov = fovs[next_index]
            fov.rank = _
            sorted_fovs.append(fov)

        return sorted_fovs

    def find_candidate_fovs(
        self, images: Optional[DataArray] = None, n_candidates: int = 50, **kwargs
    ) -> List[FocusFOV]:
        """Find and rank candidate FOVs for focusing.

        Args:
            images: HiSeqImages object, uses self.rough_ims if not provided
            n_candidates: Number of candidate FOVs to find

        Returns:
            Array of [row, col] pixel positions for candidate FOVs
        """
        if images is None:
            images = self.rough_ims

        if images is None:
            raise ValueError("No images available for FOV detection")

        LOGGER.info("Autofocus:: Finding candidate FOVs")

        # Sum channels with signal
        self.sum_image = sum_images(images)
        self.median = np.median(self.sum_image)
        self.MAD = median_abs_deviation(self.sum_image, axis=None)
        self.score_thresh = 3.5 * self.MAD**2 / self.median

        LOGGER.debug(f"Median sum image px = {self.median}")
        LOGGER.debug(f"MAD sum image px = {self.MAD}")
        LOGGER.debug(f"Score threshold = {self.score_thresh}")

        if self.sum_image is None:
            LOGGER.warning("Autofocus:: No signal in channels")
            return []

        signal_fov = []
        for fov in self.generate_mini_tiles():
            # Filter out tiles without signal
            fov = self.filter_empty_fov(fov)
            if fov is not None:
                # Find focus fov within tile
                signal_fov.append(Autofocus.refine_fov(fov))

        # Sort by FOV score
        signal_fov.sort(key=attrgetter("score"), reverse=True)
        n_fovs = len(signal_fov)

        # Select top FOVS

        if n_fovs >= n_candidates:
            signal_fov = signal_fov[0:n_candidates]

        # Sort so FOVs are spatially uniform
        if n_fovs > n_candidates / 10:
            sorted_fovs = Autofocus.spatial_uniform_sort(signal_fov)

        if n_fovs >= n_candidates:
            LOGGER.info(f"Autofocus:: Found {n_fovs} candidate FOVs")
            return sorted_fovs
        else:
            LOGGER.warning(f"Autofocus:: Only found {n_fovs} candidate FOVs")
            if n_fovs > n_candidates / 10:
                return sorted_fovs
            else:
                return signal_fov

        # # Determine scale and number of markers
        # px_rows, px_cols = self.sum_image.shape
        # n_markers = int((px_rows * px_cols * self.scale**2) ** 0.5 * 0.5)
        # n_markers = max(n_markers, n_candidates)

        # # Get focus points based on routine type
        # if "partial" in self.focus_routine:
        #     ord_points = get_focus_points_partial(self.sum_image, self.scale, n_markers)
        # else:
        #     ord_points = get_focus_points(self.sum_image, self.scale, n_markers)

        # LOGGER.info(f"Autofocus:: Found {len(ord_points)} candidate FOVs")

        # return ord_points

    async def evaluate_fov(
        self, fov: FocusFOV, roi: Optional[BaseROI] = None
    ) -> Optional[Tuple[int, int, int]]:
        """Evaluate a single FOV by taking a focus stack and finding best Z.

        Args:
            px_pt: [row, col] pixel position
            roi: Optional ROI override

        Returns:
            Tuple of (x_pos, y_pos, best_z) or None if focus not found
        """
        if roi is None:
            roi = self.roi

        # Convert pixel position to stage coordinates
        x_step, y_step = self.microscope.px_to_step(
            fov.px_row, fov.px_col, roi, self.scale
        )

        LOGGER.info(f"Autofocus:: Evaluating FOV at x={x_step}, y={y_step}")

        # Move to position
        await self.microscope._move(x=x_step, y=y_step)

        # Take focus stack
        n_frames = roi.focus.n_frames
        focus_stack = await self.microscope._focus_stack(n_frames=n_frames)

        # Store FOV dimensions from focus stack images for visualization
        if self.fov_shape is None and focus_stack.size > 0:
            self.fov_shape = (len(focus_stack.row), len(focus_stack.col))

        # Find best Z position
        fov_label = f"r{fov.px_row}_c{fov.px_col}"
        best_z = self.find_best_z(focus_stack, fov_label=fov_label)

        fov.x = x_step
        fov.y = y_step
        fov.z = best_z
        if best_z is not None:
            LOGGER.debug(
                f"Autofocus:EvaluateFocus: Found focus at x={x_step}, y={y_step}, z={best_z}"
            )
        else:
            LOGGER.debug(
                f"Autofocus:EvaluateFocus: No focus found at x={x_step}, y={y_step}"
            )
        return fov

    def format_focus_data(self, focus_stack: DataArray) -> Union[np.ndarray, bool]:
        """Format focus stack data for analysis using Tenengrad focus metric.

        Args:
            focus_stack: Focus stack from camera with dimensions (channel, z, row, col).

        Returns:
            Focus metric array of shape (n_frames, n_channels), or False if no signal.
        """
        import time

        n_channels, n_frames = focus_stack.shape[0:2]
        LOGGER.debug(
            f"Autofocus:FormatFocusData: formatting {n_channels}×{n_frames} frames"
        )
        t0 = time.perf_counter()

        if focus_stack.dtype == object:
            focus_metric = self._format_focus_data_loop(
                focus_stack, n_channels, n_frames
            )
        else:
            # Vectorized path for numeric DataArrays (n_channels, n_frames, row, col)
            arr = focus_stack.values.astype(np.float32)
            # Sobel along row axis (axis=2) replicates skimage.sobel_v per 2D frame
            gradient = ndimage_sobel(arr, axis=2)
            # Tenengrad: sum of squared gradients over spatial dims → (n_channels, n_frames)
            focus_metric = np.sum(
                gradient**2, axis=(-2, -1)
            ).T  # → (n_frames, n_channels)

        elapsed = time.perf_counter() - t0
        LOGGER.debug(f"Autofocus:FormatFocusData: done in {elapsed:.3f}s")

        if focus_metric is False or np.sum(focus_metric) == 0:
            LOGGER.debug("No signal in focus stack")
            return False

        return focus_metric

    def _format_focus_data_loop(
        self, focus_stack: DataArray, n_channels: int, n_frames: int
    ) -> Union[np.ndarray, bool]:
        """Loop-based Tenengrad metric for object-dtype DataArrays (test/synthetic stacks)."""
        focus_metric = np.zeros((n_frames, n_channels))
        for i in range(n_frames):
            for j, ch in enumerate(focus_stack.channel):
                img = focus_stack.sel(channel=ch).isel(z=i)
                if img is not None and img.shape[0] >= 3 and img.shape[1] >= 3:
                    gradient = sobel_v(img.astype(float))
                    focus_metric[i, j] = np.sum(gradient**2)
        return focus_metric

    def find_best_z(
        self, focus_stack: DataArray, fov_label: str = ""
    ) -> Union[int, float, None]:
        """Find the best Z position from a focus stack.

        Uses mixed Gaussian fitting to find the optimal focus position.

        Args:
            focus_stack: Focus stack data from Microscope._focus_stack,
                         shape (n_frames, n_channels) with dtype=object
            fov_label: Label used in the output filename when plot_data is enabled

        Returns:
            Best Z position or None if not found
        """
        if focus_stack.size == 0:
            return None

        # Format focus data for fitting
        focus_metric = self.format_focus_data(focus_stack)

        if focus_metric is False:
            return None

        # Sum across channels to get single focus metric per frame
        combined_metric = np.sum(focus_metric, axis=1)

        # Normalize
        combined_metric = combined_metric / combined_metric.max()

        # Stack steps and metrics
        formatted = np.vstack((focus_stack.z.values, combined_metric)).T

        # Fit lorentzian
        best_z, popt = self.fit_lorentzian(formatted)
        if popt is not None and getattr(self.roi.focus, "plot_data", False):
            self._plot_focus_metric(formatted, popt, best_z, fov_label)

        if best_z is None:
            return None

        ztype = type(self.microscope.ZStage.max_position)

        return ztype(best_z)

    def fit_lorentzian(
        self, focus_data: np.ndarray
    ) -> Tuple[Union[int, float, None], Optional[np.ndarray]]:
        """Fit data to lorentzian model.

        Returns:
            Tuple of (best_z, popt) where popt holds all fitted parameters,
            or (None, None) on failure, or (fallback_z, None) when fit diverges.
        """

        z, focus = focus_data[:, 0], focus_data[:, 1]
        z_min, z_max = float(z.min()), float(z.max())
        z_range = z_max - z_min

        # Initial guesses for the optimizer
        wid = min(
            self.roi.focus.tolerance * self.microscope.ZStage.spum,
            z_range * 0.5,  # clamp to half the scanned range so p0 stays within bounds
        )
        p0 = [np.max(focus), z[np.argmax(focus)], wid, np.min(focus)]

        # Constrain the fit: center must stay within the scanned z range,
        # amplitude and width must be positive, offset bounded to normalized range.
        bounds = (
            [0, z_min, 0, 0],
            [np.inf, z_max, z_range, 1.0],
        )

        # Calculate the fit
        try:
            popt, pcov, infodict, mesg, ier = curve_fit(
                lorentzian, z, focus, p0=p0, bounds=bounds, full_output=True
            )
            LOGGER.debug(f"Autofocus:FitLorentzian: Success: {mesg}")

            # Validate fit quality & Reject slope/artifact fits: no real focal peak present.
            # Returns (None, popt) so evaluate_fov can plot data but not process fov.
            amp, ctr, wid_fit, off = popt
            focus_start = self.microscope.ZStage.config.get("focus_start", -np.inf)
            focus_stop = self.microscope.ZStage.config.get("focus_stop", np.inf)
            peak_too_small = amp < 0.1
            ctr_out_of_range = not (focus_start <= ctr <= focus_stop)
            bad_width = wid_fit <= 0
            tolerance_steps = self.roi.focus.tolerance * self.microscope.ZStage.spum
            insufficient_contrast = amp < off  # baseline dominates peak signal
            width_too_large = (
                wid_fit > 3 * tolerance_steps
            )  # peak spans >> depth of field

            if (
                insufficient_contrast
                or width_too_large
                or peak_too_small
                or ctr_out_of_range
                or bad_width
            ):
                LOGGER.warning(
                    f"Autofocus:FitLorentzian: Rejecting low-quality focal point "
                    f"(amp={amp:.3f}, off={off:.3f}, ctr={ctr:.0f}, wid={wid_fit:.0f})"
                )
                return None, popt

            return ctr, popt
        except ValueError as e:
            LOGGER.error(e)
            return None, None
        except (RuntimeError, OptimizeWarning) as e:
            LOGGER.warning(e)
            LOGGER.warning(
                "Autofocus:FitLorentzian: Falling back to largest focus metric"
            )
            return z[np.argmax(focus)], None

    def _plot_focus_metric(
        self,
        focus_data: np.ndarray,
        popt: Optional[np.ndarray],
        best_z: Union[int, float, None],
        fov_label: str = "",
    ) -> None:
        """Save a PNG of the focus metric vs z position with the fitted curve."""
        z, metric = focus_data[:, 0], focus_data[:, 1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(z, metric, s=20, color="steelblue", label="measured", zorder=3)

        if popt is not None:
            z_fine = np.linspace(z.min(), z.max(), 300)
            ax.plot(
                z_fine,
                lorentzian(z_fine, *popt),
                color="tomato",
                linewidth=1.5,
                label="Lorentzian fit",
            )

        if best_z is not None:
            ax.axvline(
                best_z,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"best z = {best_z}",
            )
        ax.set_xlabel("Z position (steps)")
        ax.set_ylabel("Focus metric (normalized)")
        ax.set_title(f"Focus metric — {fov_label}" if fov_label else "Focus metric")
        ax.legend(fontsize=8)

        out_path = Path(self.focus_output) / f"focus_metric_{fov_label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.debug(f"Autofocus: Saved focus metric plot to {out_path}")

    def ransac_focus(
        self,
        focus_df: Optional[pd.DataFrame] = None,
        n_markers: int = 0,
        roi: Optional[BaseROI] = None,
    ) -> Optional[Pipeline]:
        """Use RANSAC to find consensus focal plane from multiple FOVs.

        Args:
            focus_points: Array of [x, y, z] positions

        Returns:
            Tuple of (slope, intercept) for focal plane or None if no consensus
        """

        if focus_df is None:
            focus_df = self.focus_df

        focus_df = focus_df[focus_df["z"].notna()]
        n_total = len(focus_df)
        if n_total < 3:
            LOGGER.warning("Autofocus:: Not enough points for RANSAC")
            return None

        if n_markers == 0:
            n_markers = n_total

        X = focus_df[["x", "y"]]  # x, y positions
        y = focus_df["z"]  # z positions

        # Define focus threshold
        if roi is None and self.roi is not None:
            roi = self.roi
            # Convert to steps
            threshold = roi.focus.tolerance * self.microscope.ZStage.config["spum"]
        else:
            threshold = None

        # Create polynomial features for plane fitting
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=1),
            RANSACRegressor(
                residual_threshold=threshold,  # Z-step threshold for inliers
                min_samples=3,
                random_state=42,
            ),
        )

        try:
            model.fit(X, y)

            # Get inlier mask
            inlier_mask = model.named_steps["ransacregressor"].inlier_mask_
            focus_df["inlier"] = inlier_mask
            self.focus_df.loc[focus_df.index, "inlier"] = focus_df["inlier"]
            n_inliers = np.sum(inlier_mask)

            LOGGER.debug(f"Autofocus:: RANSAC found {n_inliers}/{n_total} inliers")

            # Check if enough consensus (at least 50% inliers)
            if n_inliers < n_markers:
                LOGGER.warning("Autofocus:: Not enough consensus for focal plane")
                return None

            return model

        except Exception as e:
            LOGGER.error(f"Autofocus:: RANSAC failed: {e}")
            return None

    def map_and_save_focus_fovs(
        self, candidate_fovs: List[FocusFOV], output_path: Optional[Path] = None
    ) -> Path:
        """Map out focusing FOVs on the rough scan image and save visualization.

        Creates an annotated image showing:s
        - Candidate FOV positions as colored rectangles
        - Evaluated FOVs that are RANSAC inliers (blue) or outliers (red)
        - Unevaluated candidates colored by ranking (green=best to yellow=worst)
        - Best FOV highlighted with white border

        Args:
            candidate_fovs: Array of [row, col] pixel positions for all candidates
            output_path: Optional output path for saved image

        Returns:
            Path to saved FOV map image
        """

        # Get summed image for background
        if self.rough_ims is None:
            raise ValueError("No rough scan images loaded")

        if self.sum_image is None:
            self.sum_image = sum_images(self.rough_ims)
        if self.sum_image is None:
            raise ValueError("Could not sum channel images")

        # Default output path
        if output_path is None:
            output_path = self.focus_output / f"FOV_Map_{self.roi.name}.png"
        else:
            output_path = Path(output_path) / f"FOV_Map_{self.roi.name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get FOV dimensions from focus stack images
        if self.fov_shape is not None:
            fov_height, fov_width = self.fov_shape
        else:
            fov_height, fov_width = 100, 100
            LOGGER.warning("FOV dimensions not set, using default 100x100")

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Display summed image (grayscale)
        ax.imshow(self.sum_image, cmap="gray", aspect="auto")

        # Create colormap for ranking (green to yellow gradient)
        n_candidates = len(candidate_fovs)
        n_evaluated = len(self.focus_df)

        # Track counts for legend
        n_inliers = self.focus_df.inlier.sum()
        n_outliers = n_evaluated - n_inliers
        n_unevaluated = n_candidates - n_evaluated

        # Draw FOV rectangles
        if n_evaluated > 0:
            n_to_draw = min(n_evaluated * 2, n_candidates)
        else:
            n_to_draw = n_candidates
        # idx = 0
        for fov in candidate_fovs[0:n_to_draw]:
            row, col = fov.px_row, fov.px_col
            # fov_key = (int(row), int(col))

            # Calculate rectangle position (centered on FOV coordinates)
            rect_x = col - fov_width / 2
            rect_y = row - fov_height / 2

            # Determine color and style based on FOV status
            mask = (self.focus_df.px_row == row) & (self.focus_df.px_col == col)
            if mask.any():
                inlier = self.focus_df.loc[mask, "inlier"].values
                alpha = 0.8

                if inlier:
                    # RANSAC inlier - blue
                    color = "blue"
                    linewidth = 2.5
                else:
                    # RANSAC outlier - red
                    color = "red"
                    linewidth = 2.5
            else:
                # Unevaluated candidate - green to yellow gradient based on rank
                if n_evaluated > 0:
                    rank_ratio = (fov.rank - n_evaluated) / min(
                        n_unevaluated, n_evaluated
                    )
                else:
                    rank_ratio = fov.rank / n_unevaluated
                # Green (best) to yellow (worst)
                color = (rank_ratio, 1.0 - rank_ratio * 0.5, 0)
                linewidth = 1.5
                alpha = 0.1

            # Draw rectangle
            rect = Rectangle(
                (rect_x, rect_y),
                fov_width,
                fov_height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                alpha=alpha,
            )
            ax.add_patch(rect)

            # # Add X marker for outliers
            # if hasattr(fov, "z") and not inlier:
            #     ax.plot(col, row, "rx", markersize=12, markeredgewidth=2, alpha=0.9)

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = []

        if n_unevaluated > 0 and n_inliers == 0 and n_outliers == 0:
            # Only unevaluated candidates
            legend_elements.append(
                Patch(
                    facecolor=(0.5, 0.75, 0),
                    edgecolor="black",
                    label=f"Unevaluated candidates ({n_unevaluated})",
                )
            )
        else:
            # Show evaluated status
            if n_inliers > 0:
                legend_elements.append(
                    Patch(
                        facecolor="none",
                        edgecolor="blue",
                        label=f"RANSAC inliers ({n_inliers})",
                        linewidth=2.5,
                    )
                )
            if n_outliers > 0:
                legend_elements.append(
                    Patch(
                        facecolor="none",
                        edgecolor="red",
                        label=f"RANSAC outliers ({n_outliers})",
                        linewidth=2.5,
                    )
                )
            if n_unevaluated > 0:
                legend_elements.append(
                    Patch(
                        facecolor=(0.5, 0.75, 0),
                        edgecolor="black",
                        label=f"Not evaluated ({n_unevaluated})",
                    )
                )

        # if best_fov_idx is not None:
        #     legend_elements.append(
        #         Patch(
        #             facecolor="none", edgecolor="white", label="Best FOV", linewidth=3
        #         )
        #     )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=10,
                framealpha=0.9,
                edgecolor="black",
            )

        # Title with metadata
        roi_name = self.roi.name if hasattr(self.roi, "name") else "Unknown"
        routine = self.focus_routine
        optimal_z_str = f"{self.optimal_z}" if self.optimal_z is not None else "N/A"

        title = f"Focus FOV Map - {roi_name}\n"
        title += f"Routine: {routine} | Optimal Z: {optimal_z_str} | "
        title += f"Candidates: {n_candidates} | Evaluated: {n_evaluated}"

        if n_inliers > 0 or n_outliers > 0:
            title += f" | Inliers: {n_inliers}/{n_inliers + n_outliers}"

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Column (pixels)", fontsize=12)
        ax.set_ylabel("Row (pixels)", fontsize=12)

        # Add scale bar (optional, based on image size)
        img_height, img_width = self.sum_image.shape
        scale_bar_length = int(img_width * 0.1)  # 10% of image width
        if scale_bar_length > 0:
            ax.plot(
                [50, 50 + scale_bar_length],
                [img_height - 50, img_height - 50],
                "w-",
                linewidth=3,
            )
            ax.text(
                50 + scale_bar_length / 2,
                img_height - 40,
                f"{scale_bar_length} px",
                color="white",
                fontsize=10,
                ha="center",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save figure
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        LOGGER.info(
            f"FOV map saved: {n_inliers} inliers, {n_outliers} outliers, "
            f"{n_unevaluated} unevaluated out of {n_candidates} candidates"
        )

        return output_path

    def target_inliers(self, roi: Optional[BaseROI] = None) -> Tuple[int, BaseROI]:
        """Get number of inliers for consesus focus plane of roi."""
        if roi is None:
            roi = self.roi

        sensor_height = self.microscope.Camera.config.get("sensor_height")
        cols = abs(roi.stage.y_last - roi.stage.y_init) / self.microscope.YStage.spum
        n_per_tile = max(2, roi.focus.coverage * cols / sensor_height)

        if "full" in roi.focus.routine:
            # Scale n_inliers based on area of roi
            nx = max(2, roi.stage.nx - 1)
            n_inliers = max(5, round(nx * n_per_tile / 2))
            if n_inliers % 2 == 0:
                # Add 1 if even
                n_inliers += 1

            return n_inliers, roi

        elif "partial" in roi.focus.routine:
            # Scale n_inliers based on length of roi
            x_fraction = roi.focus.x_fraction
            if x_fraction > 1:
                x_fraction = 1

            nx = max(1, round(roi.stage.nx * x_fraction))
            x_center = roi.stage.x_center
            x_dir = roi.stage.x_direcion
            x_step = roi.stage.x_step
            x_init = int(x_center - x_dir * nx / 2 * x_step)
            x_last = int(x_center + x_dir * nx / 2 * x_step)

            n_inliers = round(n_per_tile)
            n_inliers += n_inliers % 2  # Add 1 if even

            update = {"stage": {"nx": nx, "x_init": x_init, "x_last": x_last}}
            focus_roi = roi.model_copy(update=update)

            return n_inliers, focus_roi
        else:
            # Fall back to 5 inliers
            return 5, roi

    async def run(self, roi: Optional[BaseROI] = None) -> Optional[int]:
        """Run complete autofocus routine.

        Args:
            roi: Optional ROI override

        Returns:
            Optimal Z focus position or None if autofocus failed
        """
        if roi is None:
            roi = self.roi

        LOGGER.info(f"Autofocus:: Starting autofocus {roi.focus.routine} routine")

        # Get target number of inliers and update roi for focusing
        n_markers, froi = self.target_inliers(roi)
        LOGGER.debug(f"Autofocus:: need {n_markers} FOVs for consesus focal plane")

        # Step 1: Capture rough scan
        img_dir = await self.capture_rough_scan(froi)

        # Step 2: Load and process images
        self.load_and_process_images(img_dir)

        # Step 3: Find candidate FOVs
        candidate_fovs = self.find_candidate_fovs(n_candidates=n_markers * 10)
        n_candidates = len(candidate_fovs)
        if n_candidates == 0:
            LOGGER.error("Autofocus:: No candidate FOVs found")
            return None
        elif n_candidates < n_markers:
            LOGGER.error("Autofocus:: Not enough candidate FOVs found")
            return None

        # Step 4: Iteratively evaluate FOVs
        model = None
        idx = 0
        evaluated_fovs = []
        n_fovs = n_markers
        batch_size = n_markers
        while model is None and idx < n_candidates:
            fov = candidate_fovs[idx]
            fov = await self.evaluate_fov(fov, froi)

            evaluated_fovs.append(fov)
            n_evaluted = len(self.focus_df) + len(evaluated_fovs)
            if fov.z is not None:
                LOGGER.info(f"Autofocus:: Found {n_evaluted}/{n_fovs} focus points")

            if len(evaluated_fovs) >= batch_size:
                new_points = pd.DataFrame(evaluated_fovs)
                self.focus_df = pd.concat(
                    [self.focus_df, new_points], ignore_index=True
                )
                n_good_fovs = self.focus_df["z"].notna().sum()
                # Try RANSAC with current points
                if n_good_fovs >= n_markers:
                    model = self.ransac_focus(n_markers=n_markers)

                    if model is not None:
                        LOGGER.info("Autofocus:: Found consensus focal plane")
                        valid_df = self.focus_df[self.focus_df["z"].notna()]
                        opt_z = round(model.predict(valid_df[["x", "y"]]).mean())

                    else:
                        # No consesus keep iterating through more fovs
                        evaluated_fovs = []
                        LOGGER.warning(
                            "Autofocus:: Could not find consesus focal plane"
                        )
                        if idx < n_candidates - 1:
                            batch_size = get_new_batch_size(n_good_fovs, n_markers)
                            batch_size = min(batch_size, n_candidates - idx)
                            n_fovs += batch_size
                            LOGGER.info(
                                "Autofocus:: Evaluate 2 additional focus points"
                            )
                elif idx < n_candidates - 1:
                    evaluated_fovs = []
                    batch_size = get_new_batch_size(n_good_fovs, n_markers)
                    batch_size = min(batch_size, n_candidates - idx)
                    n_fovs += batch_size

            idx += 1

        valid_z = self.focus_df["z"].dropna()
        if len(valid_z) == 0:
            LOGGER.error("Autofocus:: FAILED - Could not find any focus points")
            return None

        if model is None:
            LOGGER.warning("Autofocus:: No consesus focal plane, using median Z")
            opt_z = int(np.median(valid_z))

        # # Step 5: Calculate optimal Z from consensus
        self.optimal_z = opt_z
        LOGGER.info(f"Autofocus:: Optimal Z position: {opt_z}")

        # Move to optimal Z position
        await self.microscope.ZStage.move(opt_z)

        # Generate and save FOV map visualization
        if getattr(self.roi.focus, "plot_data", True):
            try:
                fov_map_path = self.map_and_save_focus_fovs(candidate_fovs, img_dir)
                LOGGER.info(f"Autofocus:: FOV map saved to {fov_map_path}")
            except Exception as e:
                LOGGER.warning(f"Autofocus:: Failed to save FOV map: {e}")

        return opt_z


def lorentzian(x, amp, ctr, wid, off):
    """Lorentzian model fot fitting focus data."""
    return off + amp * (1 / (1 + ((x - ctr) / wid) ** 2))


async def autofocus(microscope, roi: BaseROI) -> Optional[int]:
    """Main autofocus function.

    Args:
        microscope: Microscope instance
        roi: ROI with focus parameters

    Returns:
        Optimal Z focus position or None if failed
    """
    af = Autofocus(microscope, roi)
    return await af.run(roi)


def get_new_batch_size(n_good_fovs, n_markers):
    if n_good_fovs < n_markers:
        return n_markers - n_good_fovs
    elif n_good_fovs >= n_markers:
        return n_good_fovs % 2 + 1
