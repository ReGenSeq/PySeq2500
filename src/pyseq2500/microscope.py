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
from typing import Type, Union, Literal
from math import ceil
from functools import cached_property

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
        roi: ROIType = None,
        name: str = "",
        y_init: int = None,
        y_last: int = -1,
        n_frames: int = 0,
        **kwargs,
    ):
        """Capture an image and save it to filename."""

        # Check parameters
        # Specified parameters can override parameters stored in roi
        # If parameters not specified or not in roi default to config or current positions

        if y_init is None and roi is not None:
            y_init = roi.stage.y_init
        elif y_init is None:
            y_init = self.YStage.position

        height = self.YStage.config["step"]
        if n_frames == 0 and roi is not None:
            n_frames = roi.stage.ny
        elif n_frames == 0 and y_last < y_init:
            n_frames = ceil(y_init - y_last) / height
        else:
            raise ValueError("Number of frames, n_frames, must be > 0.")

        n_triggers = n_frames * height
        if roi is not None:
            image_dir = roi.image.output
            y_last = roi.stage.y_last
        else:
            image_dir = DEFAULT_CONFIG["experiment"]["output_path"]
            # Recalculate y_last from y_init based on number of frames
            y_delta = n_triggers * self.resolution * self.YStage.spum
            y_last = int(y_init - y_delta - 300000)

        # Check Cameras and sync YStage with FPGA
        await asyncio.gather(
            self.sync_YStage_FPGA(),
            self.Camera.check_free(),
        )
        # Allocate memory and set TDI triggering
        await asyncio.gather(
            self.Camera.allocate("TDI", n_frames), self.FPGA.TDIARM(y_init, n_triggers)
        )

        # Arm Camera triggers
        await self.FPGA.TDIARM(n_triggers, y_init)

        # Start Imaging
        await self.Camera.startAcquisition()  # Start cameras
        await self.Shutter.open()  # Open laser shutter
        await self.YStage.move(y_last)  # Move Y Stage

        # Stop Imaging
        await self.Shutter.close()  # Close laser shutter
        await self.Camera.stopAcquisition()  # Stop cameras

        # Save Images
        nbytes = await self.Camera.save_image(name, image_dir)  # Stop cameras

        return nbytes

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
            z_last = roi.stage.z_init

        if z_step == 0 and roi is not None:
            z_step = roi.stage.z_step
        else:
            z_step = self.ZStage.step

        if z_init == -1 or z_last == -1:
            raise ValueError("Specify initial and last Z stage position.")

        # Loop over z stack
        for z in range(roi.stage.z_init, roi.stage.z_last, roi.stage.z_step):
            await self.ZStage.move(z)
            await self._capture(roi, name=f"{name}_z{z}")

    @check_name
    async def _scan(self, roi: ROIType = None, name: str = "", **kwargs):
        """Perform a scan over the specified region of interest (ROI)."""

        if roi is not None:
            LOGGER.info(f"Microscope:: Moving to {roi.name} initial position")
            # Move to initial position
            y_init = roi.stage.y_init
            x_init = roi.stage.x_init
            await self._move(x=x_init, y=y_init)

        # Loop over x tiles
        _x = range(x_init, roi.stage.x_last, roi.stage.x_step)
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
