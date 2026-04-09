from typing import Optional
from attrs import field
from pyseq_core.base_system import BaseSequencer
from pyseq_core.base_protocol import CUSTOM_ROI
from pyseq2500.flowcell import FlowCell
from pyseq2500.utils import HW_CONFIG, DEFAULT_CONFIG
from math import ceil, floor


class PySeq2500(BaseSequencer):
    _flowcells: dict[str, FlowCell] = field(init=False)  # pyright: ignore[reportIncompatibleVariableOverride]

    @_flowcells.default  # pyright: ignore[reportAttributeAccessIssue]
    def set_flowcells(self):
        return {fc: FlowCell(name=fc) for fc in ["A", "B"]}

    @staticmethod
    def custom_roi_stage(roi: Optional[CUSTOM_ROI], **kwargs) -> dict:
        """Take LLx, LLy, URx, URy coordinates and return stage position parameters."""

        if roi is None:
            kwargs.setdefault("overlap", DEFAULT_CONFIG["image"]["overlap"])
            roi = CUSTOM_ROI(**kwargs)

        LLx = roi.LLx
        LLy = roi.LLy
        URx = roi.URx
        URy = roi.URy
        flowcell = roi.flowcell

        # x, y, Steps Per UMicron
        x_spum = HW_CONFIG["XStage"]["spum"]
        y_spum = HW_CONFIG["YStage"]["spum"]

        #  Convert from mm to stage steps
        LLx = LLx * 1000 * x_spum
        LLy = LLy * 1000 * y_spum
        URx = URx * 1000 * x_spum
        URy = URy * 1000 * y_spum

        # Fixed Parameters
        x_origin = HW_CONFIG["XStage"]["origin"][flowcell]  # reference x step
        y_origin = HW_CONFIG["YStage"]["origin"]  # reference y step
        resolution = HW_CONFIG["Microscope"]["resolution"]  # microns / pixel
        tile_width = HW_CONFIG["Cameras"]["sensor_width"] * resolution  # microns
        bundle_height = HW_CONFIG["Cameras"]["TDI"]["sensor_mode_line_bundle_height"]
        bundle_height = bundle_height * y_spum * resolution  # microns

        # Number of tiles
        x_step = floor(
            (tile_width - resolution * roi.overlap) * x_spum
        )  # x steps between tiles
        x_width = LLx - URx
        n_tiles = ceil(x_width / x_step)
        overlap = resolution * roi.overlap * x_spum

        # Tile positions
        x_center = int(x_origin - LLx + (x_width) / 2)
        x_init = floor(x_center - (n_tiles * x_step / 2))
        x_last = ceil(x_init + n_tiles * x_step)

        y_init = ceil(y_origin + LLy)
        y_last = floor(y_origin + URy)
        y_center = int(y_init - (y_init - y_last) / 2)

        # Number of camera frames
        n_frames = ceil((LLy - URy) / y_spum / resolution / bundle_height) + 10

        # Adjust x and y center so focus will image (32 frames, 128 bundle) in center of section
        x_center -= int(tile_width * x_spum / 2)
        y_center += int(32 / 2 * bundle_height * resolution * y_spum)

        stage = {
            "flowcell": flowcell,
            "x_init": x_init,
            "x_last": x_last,
            "x_step": x_step,
            "y_init": y_init,
            "y_last": y_last,
            "n_tiles": n_tiles,
            "n_frames": n_frames,
            "x_center": x_center,
            "y_center": y_center,
            "overlap": overlap,
        }

        stage.update(kwargs.pop("stage", {}))
        return stage
