"""Autofocus module for PySeq2500.

This module provides autofocus functionality using the Microscope class.
The autofocus routine:
1. Takes an out-of-focus scan using Microscope._scan
2. Opens images with pyseq_image.pre
3. Finds and ranks promising FOVs to focus on
4. Iteratively evaluates FOVs using focus stacks and RANSAC for consensus
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from scipy.optimize import curve_fit, OptimizeWarning
import logging

from pre.image_analysis import (
    HiSeqImages,
    get_focus_points,
    get_focus_points_partial,
    sum_images,
)
from xarray import DataArray
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from pyseq_core.base_protocol import BaseROI

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.set_loglevel("info")
matplotlib.use("Agg")  # Non-interactive backend


LOGGER = logging.getLogger("PySeq")


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

    def __init__(self, microscope, roi: BaseROI):
        """Initialize Autofocus.

        Args:
            microscope: Microscope instance
            roi: ROI with focus parameters (optics, routine, output)
        """
        self.microscope = microscope
        self.roi = roi
        self.rough_ims = None
        self.sum_image = None
        self.scale = 1  # Default scale, can be adjusted based on config
        self.focus_points = None

        # State tracking for FOV mapping
        self.evaluated_fovs = {}  # {(row, col): {'z': z_val, 'inlier': bool, 'rank': int}}
        self.fov_shape = None  # (height, width) from focus stack images
        self.optimal_z = None  # Final optimal Z position
        self.inlier_mask = None  # Boolean mask for inlier FOVs

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

        # Set Z position slightly out of focus for rough scan
        z_last = roi.stage.z_init + 1

        # Capture rough scan
        await self.microscope._scan(
            roi=roi,
            name=f"RoughScan_{roi.name}",
            image_dir=roi.focus.output,
            z_last=z_last,
        )

        return roi.focus.output

    def load_and_process_images(self, image_dir: Union[str, Path]):
        """Load and process rough scan images.

        Args:
            image_dir: Directory containing rough scan images

        Returns:
            HiSeqImages object with loaded images
        """
        image_dir = Path(image_dir)

        LOGGER.info(f"Autofocus:: Loading images from {image_dir}")

        # Load images using pyseq_image.pre
        self.rough_ims = HiSeqImages.open_RoughScan(str(image_dir), ["tif", "tiff"])

        # Correct background if method exists
        if hasattr(self.rough_ims, "correct_background"):
            self.rough_ims.correct_background()
        # Correct background if method exists
        if hasattr(self.rough_ims, "register_channels"):
            self.rough_ims.register_channels()

        LOGGER.debug(self.rough_ims.im)

        return self.rough_ims

    def find_candidate_fovs(
        self, images: Optional[HiSeqImages] = None, n_candidates: int = 50
    ) -> np.ndarray:
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
        self.sum_image = sum_images(images.im)

        if self.sum_image is None:
            LOGGER.warning("Autofocus:: No signal in channels")
            return np.array([])

        # Determine scale and number of markers
        px_rows, px_cols = self.sum_image.shape
        n_markers = int((px_rows * px_cols * self.scale**2) ** 0.5 * 0.5)
        n_markers = max(n_markers, n_candidates)

        # Get focus points based on routine type
        if "partial" in self.focus_routine:
            ord_points = get_focus_points_partial(self.sum_image, self.scale, n_markers)
        else:
            ord_points = get_focus_points(self.sum_image, self.scale, n_markers)

        LOGGER.info(f"Autofocus:: Found {len(ord_points)} candidate FOVs")

        return ord_points

    async def evaluate_fov(
        self, px_pt: np.ndarray, roi: Optional[BaseROI] = None
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
        x_pos, y_pos = self.microscope.px_to_step(px_pt[0], px_pt[1], roi, self.scale)

        LOGGER.debug(f"Autofocus:: Evaluating FOV at x={x_pos}, y={y_pos}")

        # Move to position
        await self.microscope._move(x=x_pos, y=y_pos)

        # Take focus stack
        n_frames = roi.focus.n_frames

        focus_stack = await self.microscope._focus_stack(n_frames=n_frames)

        # Store FOV dimensions from focus stack images for visualization
        if self.fov_shape is None and focus_stack.size > 0:
            self.fov_shape = focus_stack.chunks[0:2]

            # first_img = focus_stack.data[0,0]
            # if first_img is not None:
            #     self.fov_shape = first_img.shape  # (height, width)

        # Find best Z position
        best_z = self.find_best_z(focus_stack)

        if best_z is not None:
            LOGGER.debug(f"Autofocus:: Found focus at x={x_pos}, y={y_pos}, z={best_z}")
            return (x_pos, y_pos, best_z)
        else:
            LOGGER.debug(f"Autofocus:: No focus found at x={x_pos}, y={y_pos}")
            return None

    def format_focus_data(self, focus_stack: DataArray) -> Union[np.ndarray, bool]:
        """Format focus stack data for analysis using Tenengrad focus metric.

        Args:
            focus_stack: Focus stack from camera, shape (n_frames, n_channels)
                         with dtype=object, each element is a 2D image array

        Returns:
            Formatted focus data array [n_frames, n_channels] with numeric dtype
            containing focus metrics, or False if no signal
        """
        from skimage.filters import sobel_v

        n_channels, n_frames = focus_stack.shape[0:2]

        # Calculate Tenengrad focus metric per frame per channel
        # Tenengrad = sum of squared gradients (using vertical Sobel for horizontal edges)
        focus_metric = np.zeros((n_frames, n_channels))
        for i in range(n_frames):
            for j, ch in enumerate(focus_stack.channel):
                img = focus_stack.sel(channel=ch).isel(z=i)
                if img is not None and img.shape[0] >= 3 and img.shape[1] >= 3:
                    # Apply vertical Sobel filter (detects horizontal edges)
                    gradient = sobel_v(img.astype(float))
                    # Tenengrad: sum of squared gradients
                    focus_metric[i, j] = np.sum(gradient**2)

        # Check for valid signal
        if np.sum(focus_metric) == 0:
            LOGGER.debug("No signal in focus stack")
            return False

        return focus_metric

    def find_best_z(self, focus_stack: DataArray) -> Union[int, float, None]:
        """Find the best Z position from a focus stack.

        Uses mixed Gaussian fitting to find the optimal focus position.

        Args:
            focus_stack: Focus stack data from Microscope._focus_stack,
                         shape (n_frames, n_channels) with dtype=object

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
        LOGGER.debug(formatted)

        # # Fit lorentzian
        best_z = self.fit_lorentzian(formatted)

        return best_z

    def fit_lorentzian(self, focus_data: np.ndarray) -> Union[int, float, None]:
        """Fit data to lorentzian model."""

        z, focus = focus_data[:, 0], focus_data[:, 1]

        # Initial guesses for the optimizer
        wid = self.roi.focus.tolerance * self.microscope.ZStage.spum
        p0 = [np.max(focus), z[np.argmax(focus)], wid, np.min(focus)]
        # TODO add bounds

        # Calculate the fit
        try:
            popt, pcov, infodict, mesg, ier = curve_fit(
                lorentzian, z, focus, p0=p0, full_output=True
            )
            LOGGER.debug(f"Success: {mesg}")
            return popt[1]
        except ValueError as e:
            LOGGER.error(e)
            return None
        except (RuntimeError, OptimizeWarning) as e:
            LOGGER.warning(e)
            LOGGER.warning("Falling back to largest focus metric")
            return np.argmax(focus)

    def ransac_focus(self, focus_points: np.ndarray) -> Optional[Tuple[float, float]]:
        """Use RANSAC to find consensus focal plane from multiple FOVs.

        Args:
            focus_points: Array of [x, y, z] positions

        Returns:
            Tuple of (slope, intercept) for focal plane or None if no consensus
        """
        if len(focus_points) < 3:
            LOGGER.warning("Autofocus:: Not enough points for RANSAC")
            return None

        X = focus_points[:, :2]  # x, y positions
        y = focus_points[:, 2]  # z positions

        # Create polynomial features for plane fitting
        model = make_pipeline(
            PolynomialFeatures(degree=1),
            RANSACRegressor(
                residual_threshold=100,  # Z-step threshold for inliers
                min_samples=3,
                random_state=42,
            ),
        )

        try:
            model.fit(X, y)

            # Get inlier mask
            inlier_mask = model.named_steps["ransacregressor"].inlier_mask_
            n_inliers = np.sum(inlier_mask)
            n_total = len(focus_points)

            LOGGER.info(f"Autofocus:: RANSAC found {n_inliers}/{n_total} inliers")

            # Check if enough consensus (at least 50% inliers)
            if n_inliers / n_total < 0.5:
                LOGGER.warning("Autofocus:: Not enough consensus for focal plane")
                return None

            # Extract plane coefficients
            # coef = model.named_steps["polynomialfeatures"].transform(X)
            coefficients = model.named_steps["ransacregressor"].estimator_.coef_

            # Return slope and intercept
            return (coefficients[0], coefficients[-1])

        except Exception as e:
            LOGGER.error(f"Autofocus:: RANSAC failed: {e}")
            return None

    def map_and_save_focus_fovs(
        self, candidate_fovs: np.ndarray, output_path: Optional[Path] = None
    ) -> Path:
        """Map out focusing FOVs on the rough scan image and save visualization.

        Creates an annotated image showing:
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
        if self.rough_ims is None or self.rough_ims.im is None:
            raise ValueError("No rough scan images loaded")

        if self.sum_image is None:
            self.sum_image = sum_images(self.rough_ims.im)
        if self.sum_image is None:
            raise ValueError("Could not sum channel images")

        # Default output path
        if output_path is None:
            output_path = self.focus_output / f"FOV_Map_{self.roi.name}.png"
        output_path = Path(output_path)
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

        # Track counts for legend
        n_inliers = 0
        n_outliers = 0
        n_unevaluated = 0
        best_fov_idx = None

        # Find best FOV (lowest Z among evaluated)
        if self.evaluated_fovs:
            best_z = float("inf")
            for fov_key, fov_info in self.evaluated_fovs.items():
                if fov_info["z"] is not None and fov_info["z"] < best_z:
                    best_z = fov_info["z"]

        # Draw FOV rectangles
        n_to_draw = min(len(self.evaluated_fovs) * 2, len(candidate_fovs))
        idx = 0
        for px_pt in candidate_fovs[0:n_to_draw]:
            row, col = px_pt[0], px_pt[1]
            fov_key = (int(row), int(col))

            # Calculate rectangle position (centered on FOV coordinates)
            rect_x = col - fov_width / 2
            rect_y = row - fov_height / 2

            # Determine color and style based on FOV status
            if fov_key in self.evaluated_fovs:
                fov_info = self.evaluated_fovs[fov_key]

                alpha = 0.8

                if fov_info["inlier"] is True:
                    # RANSAC inlier - blue
                    color = "blue"
                    linewidth = 2.5
                    n_inliers += 1
                elif fov_info["inlier"] is False:
                    # RANSAC outlier - red
                    color = "red"
                    linewidth = 2.5
                    n_outliers += 1
                else:
                    # Evaluated but no RANSAC result - orange
                    color = "orange"
                    linewidth = 2
                    n_unevaluated += 1
            else:
                # Unevaluated candidate - green to yellow gradient based on rank
                rank_ratio = idx / max(n_to_draw - 1, 1)
                # Green (best) to yellow (worst)
                color = (rank_ratio, 1.0 - rank_ratio * 0.5, 0)
                linewidth = 1.5
                n_unevaluated += 1
                alpha = 0.1

                if idx < n_to_draw - 1:
                    idx += 1

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

            # Add X marker for outliers
            if (
                fov_key in self.evaluated_fovs
                and self.evaluated_fovs[fov_key]["inlier"] is False
            ):
                ax.plot(col, row, "rx", markersize=12, markeredgewidth=2, alpha=0.9)

            # Highlight best FOV with white border
            if best_fov_idx is not None and idx == best_fov_idx:
                rect_highlight = Rectangle(
                    (rect_x, rect_y),
                    fov_width,
                    fov_height,
                    linewidth=4,
                    edgecolor="white",
                    facecolor="none",
                    alpha=1.0,
                )
                ax.add_patch(rect_highlight)

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

        if best_fov_idx is not None:
            legend_elements.append(
                Patch(
                    facecolor="none", edgecolor="white", label="Best FOV", linewidth=3
                )
            )

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
        title += f"Candidates: {n_candidates} | Evaluated: {len(self.evaluated_fovs)}"

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

    async def run(self, roi: Optional[BaseROI] = None) -> Optional[int]:
        """Run complete autofocus routine.

        Args:
            roi: Optional ROI override

        Returns:
            Optimal Z focus position or None if autofocus failed
        """
        if roi is None:
            roi = self.roi

        LOGGER.info("Autofocus:: Starting autofocus routine")

        # Step 1: Capture rough scan
        image_dir = await self.capture_rough_scan(roi)

        # Step 2: Load and process images
        self.load_and_process_images(image_dir)

        # Step 3: Find candidate FOVs
        candidate_fovs = self.find_candidate_fovs(n_candidates=50)

        if len(candidate_fovs) == 0:
            LOGGER.error("Autofocus:: No candidate FOVs found")
            return None

        # Step 4: Iteratively evaluate FOVs
        n_markers = 7  # Number of good points needed (odd for median)
        focus_points_list = []
        points_evaluated = 0
        evaluated_indices = []  # Track which candidate indices were evaluated

        for idx, px_pt in enumerate(candidate_fovs):
            result = await self.evaluate_fov(px_pt, roi)
            points_evaluated += 1

            if result is not None:
                focus_points_list.append(result)
                evaluated_indices.append(idx)
                # Record evaluated FOV with its rank (index in candidate list)
                fov_key = (int(px_pt[0]), int(px_pt[1]))
                self.evaluated_fovs[fov_key] = {
                    "z": result[2],
                    "inlier": None,  # Will be updated after RANSAC
                    "rank": idx,
                }
                LOGGER.info(
                    f"Autofocus:: Found {len(focus_points_list)}/{n_markers} focus points"
                )

                if len(focus_points_list) >= n_markers:
                    # Try RANSAC with current points
                    focus_array = np.array(focus_points_list)
                    plane = self.ransac_focus(focus_array)

                    if plane is not None:
                        LOGGER.info("Autofocus:: Found consensus focal plane")
                        break

            # Safety limit: don't evaluate too many points
            if points_evaluated >= n_markers * 3:
                LOGGER.warning("Autofocus:: Reached max evaluation limit")
                break

        if len(focus_points_list) == 0:
            LOGGER.error("Autofocus:: FAILED - Could not find any focus points")
            return None

        # Step 5: Calculate optimal Z from consensus
        focus_array = np.array(focus_points_list)

        # Use RANSAC if we have enough points
        if len(focus_points_list) >= 3:
            plane = self.ransac_focus(focus_array)
            if plane is not None:
                # Get inlier mask and update evaluated FOVs
                focus_points_for_ransac = focus_array
                X = focus_points_for_ransac[:, :2]
                y = focus_points_for_ransac[:, 2]
                model = make_pipeline(
                    PolynomialFeatures(degree=1),
                    RANSACRegressor(
                        residual_threshold=100,
                        min_samples=3,
                        random_state=8387,
                    ),
                )
                model.fit(X, y)
                self.inlier_mask = model.named_steps["ransacregressor"].inlier_mask_

                # Update inlier status for evaluated FOVs
                for i, idx in enumerate(evaluated_indices):
                    if i < len(candidate_fovs):
                        px_pt = candidate_fovs[idx]
                        fov_key = (int(px_pt[0]), int(px_pt[1]))
                        if fov_key in self.evaluated_fovs:
                            self.evaluated_fovs[fov_key]["inlier"] = bool(
                                self.inlier_mask[i]
                            )

                # Calculate Z at center of ROI
                x_center = (
                    roi.stage.x_init
                    if hasattr(roi.stage, "x_init")
                    else np.mean(focus_array[:, 0])
                )
                # y_center = (
                #     roi.stage.y_init
                #     if hasattr(roi.stage, "y_init")
                #     else np.mean(focus_array[:, 1])
                # )
                slope, intercept = plane
                opt_z = int(slope * x_center + intercept)
            else:
                # Fall back to median
                opt_z = int(np.median(focus_array[:, 2]))
        else:
            # Not enough points for RANSAC, use median
            opt_z = int(np.median(focus_array[:, 2]))

        # Store optimal Z
        self.optimal_z = opt_z
        LOGGER.info(f"Autofocus:: Optimal Z position: {opt_z}")

        # Move to optimal Z position
        await self.microscope.ZStage.move(opt_z)

        # Generate and save FOV map visualization
        try:
            fov_map_path = self.map_and_save_focus_fovs(candidate_fovs)
            LOGGER.info(f"Autofocus:: FOV map saved to {fov_map_path}")
        except Exception as e:
            LOGGER.warning(f"Autofocus:: Failed to save FOV map: {e}")

        return opt_z

    # def px_to_step(
    #     self, px_row: int, px_col: int, stage_info: dict, scale: int = 1
    # ) -> Tuple[int, int]:
    #     """Convert pixel coordinates to stage step coordinates.

    #     Args:
    #         px_row: Pixel row
    #         px_col: Pixel column
    #         stage_info: Stage position information from ROI
    #         scale: Image scale factor

    #     Returns:
    #         Tuple of (x_step, y_step)
    #     """
    #     # Get reference positions
    #     x_init = stage_info.get("x_init", 0)
    #     y_init = stage_info.get("y_init", 0)

    #     # Pixel to step conversion
    #     # These values should come from microscope/camera configuration
    #     px_to_um = 0.375 * scale  # um per pixel (resolution)
    #     um_to_step_y = 1 / 0.2  # steps per um for Y stage (example)
    #     um_to_step_x = 1 / 0.1  # steps per um for X stage (example)

    #     # Calculate offsets
    #     x_offset = px_col * px_to_um * um_to_step_x
    #     y_offset = px_row * px_to_um * um_to_step_y

    #     x_step = int(x_init + x_offset)
    #     y_step = int(y_init - y_offset)  # Y is typically inverted

    #     return (x_step, y_step)


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
