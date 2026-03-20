from pyseq_core.base_system import BaseMicroscope, check_name
from pyseq_core.base_protocol import ROIFactory, OpticsParams

from pyseq2500.xstage import XStage
from pyseq2500.ystage import YStage
from pyseq2500.zstage import ZStage
from pyseq2500.tiltstage import TiltStage
from pyseq2500.laser import Laser
from pyseq2500.optics import FilterWheel, EmissionFilter, Shutter
from pyseq2500.camera import TDICameras
from pyseq2500.fpga import FPGA
from pyseq2500.com import COM_DICT
from pyseq2500.utils import DEFAULT_CONFIG

import logging
import asyncio
from attrs import define, field
from typing import Type, Union, Literal, Optional
from math import ceil
from functools import cached_property
from pathlib import Path
import numpy as np

LOGGER = logging.getLogger("PySeq")

ROI = ROIFactory.factory(DEFAULT_CONFIG)
ROIType = Type[ROI]


@define
class Microscope(BaseMicroscope):
    instruments: dict = field(init=False)
    overlap: Union[int, float] = field(default=0)

    @instruments.default
    def set_instruments(self):
        _lasers = {}
        _ex_filters = {}
        for color in ["Green", "Red"]:
            _color = color.lower()
            name = f"{color}Laser"
            _lasers[_color] = Laser(name=name, com=COM_DICT[name], color=_color)
            name = f"{color}FilterWheel"
            _ex_filters[_color] = FilterWheel(name=name, com=COM_DICT[name])

        instruments = {
            "FPGA": FPGA(com=COM_DICT["FPGA"]),
            "XStage": XStage(name="XStage", com=COM_DICT["XStage"]),
            "YStage": YStage(name="YStage", com=COM_DICT["YStage"]),
            "ZStage": ZStage(name="ZStage", com=COM_DICT["ZStage"]),
            "TiltStage": TiltStage(name="TiltStage", com=COM_DICT["TiltStage"]),
            "GreenLaser": Laser(
                name="GreenLaser", com=COM_DICT["GreenLaser"], color="green"
            ),
            "RedLaser": Laser(name="RedLaser", com=COM_DICT["RedLaser"], color="red"),
            "GreenFilterWheel": FilterWheel(
                name="GreenFilterWheel", com=COM_DICT["GreenFilterWheel"]
            ),
            "RedFilterWheel": FilterWheel(
                name="RedFilterWheel", com=COM_DICT["RedFilterWheel"]
            ),
            "EmissionFilter": EmissionFilter(
                name="EmissionFilter", com=COM_DICT["EmissionFilter"]
            ),
            "Shutter": Shutter(name="Shutter", com=COM_DICT["Shutter"]),
            "Camera": TDICameras(),
        }

        return instruments

    @property
    def Camera(self) -> TDICameras:
        """Property for the EmissionFilter."""
        return self.instruments.get("Camera")

    @property
    def EmissionFilter(self) -> EmissionFilter:
        """Property for the EmissionFilter."""
        return self.instruments.get("EmissionFilter")

    @property
    def FPGA(self) -> FPGA:
        """Property for the FPGA."""
        return self.instruments.get("FPGA")

    @cached_property
    def resolution(self) -> float:
        return self.config["resolution"]

    async def _configure(self, exp_config: dict = {}):
        """Configure microscope"""

        for instrument in self.instruments.values():
            await instrument.configure(exp_config)

    async def sync_YStage_FPGA(self):
        while abs(self.YStage.position - await self.FPGA.read_position()) > 5:
            LOGGER.debug("Microscope:: Syncing FPGA TDI to stage")
            await self.FPGA.write_position(self.YStage.position)

        LOGGER.debug("Microscope:: FPGA TDI and stage are synced")

    @check_name
    async def _capture(
        self,
        roi: Optional[ROIType] = None,
        name: str = "",
        y_init: Optional[int] = None,
        y_last: Optional[int] = None,
        n_frames: int = 0,
        reset_y: bool = False,
        image_dir: Union[str, Path] = ".",
        **kwargs,
    ):
        """Capture an image and save it to filename."""

        # Check parameters, roi, n_frames, or y_last must be specified
        # Specified parameters can override parameters stored in roi
        # If parameters not specified or not in roi default to config or current positions

        # Check if minimal args exist
        if y_last is None and n_frames == 0 and roi is None:
            raise ValueError("Must specify y_last or n_frames")

        # Get ROI (preferred)
        if n_frames == 0 and y_last is None and roi is not None:
            y_init = roi.stage.y_init
            y_last = roi.stage.y_last - 300000
            n_frames = roi.stage.n_frames
            image_dir = roi.image.image_dir

        # Get y_init
        if y_init is None:
            y_init = self.YStage.position
            if y_last is not None and y_init < y_last:
                raise ValueError("y_last must be < y_init")

        # Fall back to n_frames or y_last
        height = self.Camera.config["TDI"]["sensor_mode_line_bundle_height"]
        if n_frames == 0 and y_last is not None:
            n_frames = ceil(y_init - y_last) / height
        elif n_frames > 0 and y_last is None:
            y_last = y_init - n_frames * height - 300000
        n_triggers = n_frames * height

        # Check Cameras and sync YStage with FPGA
        await asyncio.gather(
            self.sync_YStage_FPGA(),
            self.Camera.check_free(),
        )
        # Allocate memory and set TDI triggering
        await asyncio.gather(
            self.Camera.allocate("TDI", n_frames),
            self.FPGA.TDIARM(y_init, n_triggers),
            self.YStage.set_mode("imaging"),
        )

        # Start Imaging
        await self.Camera.startAcquisition()  # Start cameras
        await self.Shutter.open()  # Open laser shutter
        await self.YStage.move(y_last)  # Move Y Stage

        # Stop Imaging
        await self.Shutter.close()  # Close laser shutter
        await self.Camera.stopAcquisition()  # Stop cameras

        # Save Images
        frame_count = await self.Camera.getFrameCount()  # Stop cameras
        await self.Camera.save_image(name, image_dir)  # Stop cameras

        # Reset for next image
        _ = [self.YStage.set_mode("moving"), self.Camera.freeFrames()]
        if reset_y:
            _.append(self.YStage.move(y_init))
        await asyncio.gather(*_)

        return frame_count == n_frames

    @check_name
    async def _z_stack(
        self,
        roi: ROIType = None,
        name: str = "",
        z_init: int = -1,
        z_last: int = -1,
        z_step: int = 0,
        **kwargs,
    ):
        """Perform a z-stack acquisition."""

        if z_init == -1 and roi is not None:
            z_init = roi.stage.z_init

        if z_last == -1 and roi is not None:
            z_last = roi.stage.z_last

        if z_step == 0 and roi is not None:
            z_step = roi.stage.z_step
        else:
            z_step = self.ZStage.step

        if z_init == -1 or z_last == -1:
            raise ValueError("Specify initial and last Z stage position.")

        # Loop over z stack
        z_pos = range(roi.stage.z_init, z_last, z_step)
        nz = len(z_pos)
        for i, z in enumerate(z_pos):
            await self.ZStage.move(z)
            reset_y = i < nz - 1
            await self._capture(roi, name=f"{name}_z{z}", reset_y=reset_y, **kwargs)

    @check_name
    async def _scan(self, roi: ROIType = None, name: str = "", **kwargs):
        """Perform a scan over the specified region of interest (ROI)."""

        if roi is not None:
            LOGGER.info(f"Microscope:: Moving to {roi.name} initial position")
            # Move to initial position
            y_init = roi.stage.y_init
            x_init = roi.stage.x_init
            x_last = roi.stage.x_last
            x_step = roi.stage.x_step
            await self._move(x=x_init, y=y_init)

        # Loop over x tiles
        _x = range(x_init, x_last, x_step)
        nx = len(_x)
        for i, x in enumerate(_x):
            LOGGER.info(f"Scanning {roi.name} tile {i + 1}/{nx}")
            await self.XStage.move(x)
            await self._z_stack(roi, name=f"s{name}_x{x}", **kwargs)

    async def _expose_scan(self, roi: ROIType, duration: Union[float, int]):
        """Scan over the specified region of interest (ROI) with laser."""
        pass

    async def _move(
        self,
        x: int = -1,
        y: Union[int, None] = None,
        z: int = -1,
        tilt1: int = -1,
        tilt2: int = -1,
        tilt3: int = -1,
        tilt: int = -1,
    ):
        """Move the stage to x,y,z,tilt coordinates."""

        msg = "Moving to"
        _ = []
        if x > 0:
            _.append(self.XStage.move(x))
            msg += f" x={x}"
        if y is not None:
            _.append(self.YStage.move(y))
            msg += f" y={y}"
        if z > 0:
            _.append(self.ZStage.move(z))
            msg += f" z={z}"

        if tilt > 0:
            tilt1 = tilt
            tilt2 = tilt
            tilt3 = tilt

        if any([tilt1 > 0, tilt2 > 0, tilt3 > 0]):
            _.append(self.TiltStage.move([tilt1, tilt2, tilt3]))
            msg += f" tilt1={tilt1} tilt2={tilt2} tilt3={tilt3}"

        LOGGER.debug(msg)
        await asyncio.gather(*_)

    async def _focus_stack(
        self, z_init: int, z_last: int, n_frames: int = 0, velocity: float = 0.0
    ):
        if n_frames == 0:
            n_frames = DEFAULT_CONFIG["focus"]["n_frames"]
        if velocity == 0.0:
            velocity = DEFAULT_CONFIG["focus"]["velocity"]

        # Setup camera, move objective to initial position and set velocity
        # of objective to move during capture.
        setup = []
        setup.append(self.Camera.check_free())
        setup.append(self.Camera.allocate("AREA", n_frames))
        setup.append(self.ZStage.set_velocity(self.ZStage.max_velocity))
        setup.append(self.ZStage.move(z_init))
        await asyncio.gather(*setup)
        await self.ZStage.set_velocity(velocity)

        # Update limits that were previously based on estimates
        # obj.update_focus_limits(cam_interval = cam1.getFrameInterval(),
        #                         range = obj.focus_range,
        #                         spacing = obj.focus_spacing)

        # Start acquiring focus stack
        await self.ZStage.set_trigger(z_init)
        await self.Shutter.open()
        try:
            start = [self.ZStage.move(z_last), self.Camera.startAcquisition()]
            await asyncio.gather(*start)
        except asyncio.TimeoutError:
            LOGGER.warning("Objective took too long to move.")

        # Wait for imaging
        try:
            await self.Camera.waitForFrames(n_frames)
        except asyncio.TimeoutError:
            LOGGER.warning("Cameras took too long to acquire frames.")

        # Stop acquiring focus stack
        stop = [self.Shutter.close(), self.Camera.stopAcquisition()]
        await asyncio.gather(*stop)

        # Get focus stack
        frame_count = await self.Camera.getFrameCount()
        if frame_count != n_frames:
            LOGGER.warning("Cameras did not acquire expected number of frames.")
            focus_stack = np.zeros((0, 0))  # Return empty array if frames not acquired
        else:
            focus_stack = await self.Camera.getFocusStack()

        await self.Camera.freeFrames()  # Free camera memory

        return focus_stack

    async def _set_parameters(
        self, image_params: OpticsParams, mode: Literal["image", "focus", "expose"]
    ):
        """Async set the parameters for the ROI."""
        pass

    async def _find_focus(self, roi: ROIType):
        """Async set the parameters for the ROI."""
        # Reset X & Y stage to initial position after finding focus
        # Save Z stage focus position to `ROI.focus.z_focus`
        # Move Z stage to `ROI.focus.z_focus`
        pass
