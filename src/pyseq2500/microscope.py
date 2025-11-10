from pyseq_core.base_system import BaseMicroscope, ROIType
from pyseq_core.base_protocol import BaseOpticsParams
import logging
import asyncio
from attrs import define, field
from typing import Literal

LOGGER = logging.getLogger("PySeq")


@define
class Microscope(BaseMicroscope):
    name: str = field(default="microscope")
    instruments: dict = field(init=False)

    @instruments.default
    def set_instruments(self):
        instruments = {}
        return instruments

    async def _configure(self, exp_config):
        LOGGER.debug(f"Configure {self.name}")

    async def _capture(self, roi: ROIType, im_name: str):
        """Capture an image and save it to the specified filename."""

        LOGGER.debug(f"Acquire {im_name}")
        await self.Shutter.move(open=True)
        await self.YStage.move(roi.stage.y_last)
        await self.Shutter.move(open=False)
        _ = []
        for c in self.Camera.values():
            _.append(c.save_image(im_name))
        _.append(self.YStage.move(roi.stage.y_init))
        await asyncio.gather(*_)

    async def _z_stack(self, roi: ROIType, im_name: str):
        """Perform a z-stack acquisition."""

        direction = roi.stage.z_direction
        step = roi.stage.z_step
        if direction == 1:
            z_init = roi.stage.z_init
            z_last = roi.stage.z_last
        else:
            z_init = roi.stage.z_last
            z_last = roi.stage.z_init
        LOGGER.debug(f"Z stack {roi.name} {z_init} to {z_last} in {step} steps")
        for i, z in enumerate(range(z_init, z_last, direction * step)):
            LOGGER.debug(f"Z stack {i}/{roi.stage.nz}")
            await self.ZStage.move(z)
            await self._capture(roi, f"{im_name}_z{z}")

    async def _scan(self, roi: ROIType, im_name: str = ""):
        """Perform a scan over the specified region of interest (ROI)."""

        x_init = roi.stage.x_init
        x_last = roi.stage.x_last
        x_step = roi.stage.x_step * roi.stage.x_direction
        LOGGER.debug(
            f"Scanning {roi.name}: XStage: {x_init} to {x_last} in {x_step} increments"
        )
        if len(im_name) == 0:
            im_name = roi.name
        for i, x in enumerate(range(x_init, x_last, x_step)):
            LOGGER.debug(f"XStage {i}/{roi.stage.nx}")
            await self.XStage.move(x)
            await self._z_stack(roi, f"{im_name}_x{x}")

    async def _expose_scan(self, roi: ROIType):
        """Async expose the sample for a specified duration without imaging."""

        x_init = roi.stage.x_init
        x_last = roi.stage.x_last
        x_step = roi.stage.x_step * roi.stage.x_direction
        n_exposures = roi.expose.n_exposures
        LOGGER.debug(
            f"Exposing {roi.name}: XStage: {x_init} to {x_last} in {x_step} increments"
        )
        for i, x in enumerate(range(x_init, x_last, x_step)):
            LOGGER.debug(f"XStage {i}/{roi.stage.nx}")
            await self.XStage.move(x)
            for n in range(n_exposures):
                LOGGER.debug(f"Exposure {n}/{n_exposures}")
                if self.YStage.position == roi.y_init:
                    await self.YStage.move(roi.y_last)
                else:
                    await self.YStage.move(roi.y_init)

    async def _find_focus(self, roi):
        LOGGER.debug(f"Fake finding focus using routine {roi.focus.routine}.")
        LOGGER.debug(f"Saving focus data to {roi.focus.output}.")
        roi.focus.z_focus = 0

    async def _move(self, roi: ROIType):
        """Move the stage ROI x,y,z coordinates."""
        LOGGER.debug(f"Moving to x={roi.x}, y={roi.y}, z={roi.z}")
        await asyncio.gather(
            self.XStage.move(roi.x),
            self.YStage.move(roi.y),
            self.ZStage.move(roi.z),
        )

    async def _set_parameters(
        self, params: BaseOpticsParams, mode: Literal["image", "focus", "expose"]
    ):
        """Set the parameters to expose/image the ROI."""

        params = params.model_dump()[mode]["optics"]
        _ = []
        for color in ["red", "green"]:
            _.append(self.Laser[color].set_power(params["power"][color]))
            _.append(self.FilterWheel[color].set_filter(params["filter"][color]))

        if mode in ["image", "focus"]:
            for c in self.Camera:
                _.append(self.Camera[c].set_exposure(params["exposure"][c]))
        await asyncio.gather(*_)
