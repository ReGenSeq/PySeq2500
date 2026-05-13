from pyseq_core.base_system import BaseMicroscope, check_name
from pyseq_core.base_protocol import BaseROI, OpticsParams

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
from pyseq2500.autofocus import autofocus

from pre.image_analysis import HiSeqImages
from xarray import DataArray

import logging
import asyncio
from attrs import define, field
from typing import Union, Optional, Tuple
from math import floor
from pathlib import Path
import numpy as np

LOGGER = logging.getLogger("PySeq")


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
            "Lasers": {
                "green": Laser(
                    name="GreenLaser", com=COM_DICT["GreenLaser"], color="green"
                ),
                "red": Laser(name="RedLaser", com=COM_DICT["RedLaser"], color="red"),
            },
            "FilterWheels": {
                "green": FilterWheel(
                    name="GreenFilterWheel", com=COM_DICT["GreenFilterWheel"]
                ),
                "red": FilterWheel(
                    name="RedFilterWheel", com=COM_DICT["RedFilterWheel"]
                ),
            },
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

    # @cached_property
    # def resolution(self) -> float:
    #     return self.config["resolution"]

    async def _configure(self, exp_config: dict = {}):
        """Configure microscope"""

        for instrument in self.iter_instruments:
            await instrument.configure(exp_config)

    async def sync_YStage_FPGA(self):
        while abs(self.YStage.position - await self.FPGA.read_position()) > 5:
            LOGGER.debug("Microscope:: Syncing FPGA TDI to stage")
            await self.FPGA.write_position(self.YStage.position)

        LOGGER.debug("Microscope:: FPGA TDI and stage are synced")

    @check_name
    async def _capture(
        self,
        roi: Optional[BaseROI] = None,
        name: str = "",
        # y_init: Optional[int] = None,
        # y_last: Optional[int] = None,
        n_frames: int = 0,
        reset_y: bool = False,
        image_dir: Union[str, Path] = ".",
    ):
        """Capture an image and save it to filename."""

        # Check parameters, roi, n_frames, or y_last must be specified
        # Specified parameters can override parameters stored in roi
        # If parameters not specified or not in roi default to config or current positions

        # Check if minimal args exist
        if n_frames == 0 and roi is None:
            raise ValueError("Must specify n_frames or roi")

        # Get y_init, ROI overrides y_init kw_arg
        y_init = self.YStage.position
        if n_frames == 0 and roi is not None:
            n_frames = roi.stage.n_frames

        # Get image_dir, image_dir kw_arg overides ROI
        if image_dir == "." and roi is not None:
            image_dir = roi.image.image_dir

        # Calculate n_triggers and y_last
        y_spum = self.YStage.config["spum"]
        height = self.Camera.config["TDI"]["sensor_mode_line_bundle_height"]
        n_triggers = n_frames * height  # px
        y_last = floor(y_init - n_triggers * self.resolution * y_spum - 300000)

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
        _ = []
        _.append(self.Shutter.close())  # Close laser shutter
        _.append(self.Camera.stopAcquisition())  # Stop cameras
        await asyncio.gather(*_)

        # Save Images
        frame_count = await self.Camera.getFrameCount()  # Stop
        if frame_count > 0:
            await self.Camera.save_image(name, image_dir)  # Stop cameras
        else:
            LOGGER.error("Cameras did not acquire frames")

        # Reset for next image
        _ = []
        _.append(self.Camera.freeFrames())
        if reset_y:
            _.append(self.YStage.move(y_init))
        _.append(self.FPGA.command("TDICLINES"))
        _.append(self.FPGA.command("TDIPULSES"))
        await asyncio.gather(*_)

        return frame_count == n_frames

    def tilt_stack(self, roi: BaseROI) -> int:
        """Acquire tilt stack from the specified region of interest (ROI)."""
        description = f"Image tilt stage stack of {roi.name}"
        return self.add_task(description, self._tilt_stack, roi)

    @check_name
    async def _tilt_stack(
        self,
        roi: Optional[BaseROI] = None,
        name: str = "",
        tilt_init: int = 0,
        tilt_last: int = 0,
        tilt_step: int = 1,
        **kwargs,
    ):
        """Scan over ROI at different tilt stage positions"""

        if roi is None:
            roi = BaseROI(**kwargs)

        LOGGER.info(f"Microscope:: Moving to {roi.name} initial position")
        # Move to initial position
        y_init = roi.stage.y_init
        x_init = roi.stage.x_init
        x_last = roi.stage.x_last
        x_step = roi.stage.x_step
        await self._move(x=x_init, y=y_init)

        # Loop over tilt position
        tilt_pos = range(tilt_init, tilt_last, tilt_step)
        nt = len(tilt_pos)
        for i, _t in enumerate(tilt_pos):
            await self.TiltStage.move(_t)
            LOGGER.info(f"Scanning {name} tilt position {_t} ({i + 1}/{nt})")
            _x = range(x_init, x_last, x_step)
            nx = len(_x)
            for ii, x in enumerate(_x):
                await self.XStage.move(x)
                _name = f"s{name}_x{x}_z{_t}"
                reset_y = ii < nx - 1 or i < nt - 1
                await self._capture(roi, _name, reset_y=reset_y, **kwargs)

    @check_name
    async def _z_stack(
        self,
        roi: Optional[BaseROI] = None,
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
            z_last = roi.stage.z_last + 1

        if z_step == 0 and roi is not None:
            z_step = roi.stage.z_step
        else:
            z_step = self.ZStage.step

        if z_init == -1 or z_last == -1:
            raise ValueError("Specify initial and last Z stage position.")

        # Loop over z stack
        z_pos = range(z_init, z_last, z_step)
        nz = len(z_pos)
        for i, z in enumerate(z_pos):
            await self.ZStage.move(z)
            reset_y = i < nz - 1
            await self._capture(roi, name=f"{name}_z{z}", reset_y=reset_y, **kwargs)

    @check_name
    async def _scan(self, roi: Optional[BaseROI] = None, name: str = "", **kwargs):
        """Perform a scan over the specified region of interest (ROI)."""

        if roi is None:
            roi = BaseROI(**kwargs)

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
            LOGGER.info(f"Scanning {name} tile {i + 1}/{nx}")
            await self._move(x=x, y=y_init)
            await self._z_stack(roi, name=f"s{name}_x{x}", **kwargs)

    async def _expose_scan(self, roi: BaseROI, duration: Union[float, int]):
        """Scan over the specified region of interest (ROI) with laser."""
        pass

    async def _move(
        self,
        x: int = -1,
        y: Union[int, None] = None,
        z: Union[int, None] = None,
        tilt1: int = -1,
        tilt2: int = -1,
        tilt3: int = -1,
        tilt: int = -1,
        **kwargs,
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
        if z is not None:
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

    def focus_z_positions(self, z_init: int, z_last: int, n_frames: int) -> np.ndarray:
        """Calculate the z position of each frame for a focus stack."""

        velocity = self.ZStage.velocity  # mm/s
        spum = self.ZStage.spum  # step/um

        velocity = velocity * 1000 * spum  # step/s
        for cam in self.Camera:
            interval = cam.getFrameInterval()  # s/frame
            z = np.arange(n_frames) * interval * velocity  # step
            z_pos = (z + z_init).clip(max=z_last)
            break

        return z_pos

    async def _focus_stack(
        self,
        z_init: Optional[int] = None,
        z_last: Optional[int] = None,
        n_frames: int = 0,
        velocity: float = 0.0,
    ) -> DataArray:
        if n_frames == 0:
            n_frames = DEFAULT_CONFIG["focus"]["n_frames"]
        if velocity == 0.0:
            velocity = DEFAULT_CONFIG["focus"]["velocity"]
        if z_init is None or z_init < self.ZStage.config["focus_start"]:
            z_init = self.ZStage.config["focus_start"]
        if z_last is None or z_last < self.ZStage.config["focus_stop"]:
            z_last = self.ZStage.config["focus_stop"]

        # Setup camera, move objective to initial position and set velocity
        # of objective to move during capture.
        setup = []
        setup.append(self.Camera.check_free())
        setup.append(self.Camera.allocate("AREA", n_frames))
        setup.append(self.ZStage.set_velocity(velocity))
        setup.append(self.ZStage.move(self.ZStage.min_position))
        await asyncio.gather(*setup)
        z_pos = self.focus_z_positions(z_init, z_last, n_frames)

        # Start acquiring focus stack
        await self.ZStage.set_trigger(z_init)
        await self.Shutter.open()
        try:
            start = [
                self.Camera.startAcquisition(),
                self.ZStage.move(self.ZStage.max_position, read=2),
            ]
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

        hs_focus_stack = HiSeqImages.open_objstack(focus_stack, z=z_pos)
        hs_focus_stack.correct_background()

        return hs_focus_stack.im

    async def _set_parameters(self, params: OpticsParams):
        """Async set the parameters for the ROI."""

        params = params.model_dump()

        _ = []
        for color in self.Lasers:
            _.append(self.Lasers[color].set_power(params["power"][color]))
            _.append(self.FilterWheels[color].set_filter(params["filter"][color]))
        if "exposure" in params:
            _.append(self.Camera.set_exposure(params["exposure"]))
        await asyncio.gather(*_)

    async def _find_focus(self, roi: BaseROI) -> BaseROI:
        """Async set the parameters for the ROI."""

        if "once" not in roi.focus.routine or roi.focus.z_focus is None:
            roi.focus.z_focus = await autofocus(self, roi)
            roi.stage.z_init = roi.focus.z_focus - roi.stage.z_step // 2 * roi.image.nz
        elif "once" in roi.focus.routine and roi.focus.z_focus is not None:
            LOGGER.debug(f"Using previous focus position {roi.focus.z_focus}")
        await self._move(x=roi.stage.x_init, y=roi.stage.y_init, z=roi.stage.z_init)
        return roi

    def px_to_step(
        self, px_row: int, px_col: int, roi: BaseROI, scale: int = 1
    ) -> Tuple[int, int]:
        """Convert pixel coordinates to stage step coordinates.

        Args:
            px_row: Pixel row
            px_col: Pixel column
            stage_info: Stage position information from ROI
            scale: Image scale factor

        Returns:
            Tuple of (x_step, y_step)
        """
        # Get reference positions
        x_init = roi.stage.x_init
        y_init = roi.stage.y_init

        # Scaled resolutio in microns per pixel
        px_to_um = self._config["resolution"] * scale  # um per pixel (resolution)

        # Calculate offsets
        x_offset = px_col * px_to_um * self.XStage.config["spum"]
        y_offset = px_row * px_to_um * self.YStage.config["spum"]

        x_step = int(x_init + x_offset)
        y_step = int(y_init - y_offset)  # Y is typically inverted

        return (x_step, y_step)
