import logging
from pyseq_core.base_instruments import BaseCamera
from attrs import define, field
from typing import Union, List
from pathlib import Path
import warnings
import asyncio
import multiprocessing

LOGGER = logging.getLogger("PySeq")

STATUS = {0: "error", 1: "busy", 2: "ready", 3: "stable", 4: "unstable"}


@define
class EmulatedTDICamera:
    camera_id: int = field()
    exposure: float = field(default=1.0)
    gain: int = field(default=1.0)
    status: int = field(default=3)
    properties: dict = field(factory=dict)
    sensor_mode: str = field(default="TDI")
    number_image_buffers: int = field(default=0)
    left_emission: str = field(init=False)
    right_emission: str = field(init=False)

    def saveImage(self, image_name, image_path):
        LOGGER.debug("Saving {image_path}/c{self.cam.left_emssion}_{image_name}.tiff")
        LOGGER.debug("Saving {image_path}/c{self.cam.right_emssion}_{image_name}.tiff")
        return 1

    def setPropertyValue(self, property, value):
        self.properties[property] = value

    def getPropertyValue(self, property):
        if property in self.properties:
            return self.properties[property]
        else:
            self.properties[property] = 1.0
            return 1.0

    def get_status(self):
        return 3

    def shutdown(self):
        pass

    def setTDI(self):
        self.setPropertyValue("sensor_mode", 4)
        self.setPropertyValue("sensor_mode_line_bundle_height", 128)
        self.sensor_mode = "TDI"

    def setAREA(self):
        self.setPropertyValue("sensor_mode", 1)
        self.setPropertyValue("sensor_mode_line_bundle_height", 128)
        self.sensor_mode = "AREA"

    def captureSetup(self):
        pass

    def stopAcquisition(self):
        pass

    def startAcquisition(self):
        pass

    def freeFrames(self):
        self.number_image_buffers = 0

    def allocFrame(self, n_frames):
        self.number_image_buffers = n_frames

    def getFrameCount(self) -> int:
        return self.number_image_buffers

    def getFocusStack(self) -> list:
        return [
            1,
        ]


def _import_dcam():
    """function to test import dcam

    dcam has a known bug with Windows.
    It sometimes can't be initialized after a period of inactivity.

    import (& initialize) dcam asynchronously with a timeout using this function.
    """


def _manage_dcam_import():
    """Run _image_dcam in background process that times out after 60 s."""
    p = multiprocessing.Process(target=_import_dcam)
    p.start()
    p.join(timeout=60)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("DCAM initialization timed out")
        return False
    return True


@define(kw_only=True)
class TDICameras(BaseCamera):
    """Wrapper around 2 instances of pyseq_core.dcam.HammatsuCamera."""

    name: str = field(default="Cameras")
    _cams: dict = field(factory=dict)
    _status: list = field(default=[None, None])
    _exposure: list = field(default=[None, None])
    _gain: list = field(default=[None, None])
    dcam_initialized: bool = field(default=True)

    @property
    def cams(self) -> dict:
        return self._cams

    def __iter__(self):
        for cam in self._cams.values():
            yield cam

    async def initialize(self):
        """Initialize cameras to TDI mode."""
        for cam in self:
            cam.setTDI()
            cam.captureSetup()

        await self.status()
        await self.get_exposure()
        await self.get_gain()

    async def configure(self):
        """Connect to cameras and configure channel names."""

        # Import dcam in seperate process asynchronously with timeout to avoid hangs
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)
            try:
                loop = asyncio.get_running_loop()
                self.dcam_initialized = True
                dcam_found = await loop.run_in_executor(None, _manage_dcam_import)
            except TimeoutError:
                LOGGER.error("DCAM initialization timed out, restart computer")
                self.dcam_initialized = False
                dcam_found = False
            except UserWarning:
                dcam_found = False

        for i in range(2):
            if i not in self.cams:
                if dcam_found:
                    # Need to reimport because test import is in different memory space
                    from pyseq_core.dcam import HamamatsuCamera

                    self.cams[i] = HamamatsuCamera(i)
                else:
                    self.cams[i] = EmulatedTDICamera(i)

        for cam in self:
            if self.config["save_nt"]:
                cam.left_emission = self.config[cam.camera_id]["nucleotides"][0]
                cam.right_emission = self.config[cam.camera_id]["nucleotides"][1]
            else:
                cam.left_emission = self.config[cam.camera_id]["left_emission"]
                cam.right_emission = self.config[cam.camera_id]["right_emission"]

    async def shutdown(self):
        """Shutdown cameras"""
        for cam in list(self.cams.keys()):
            self.cams[cam].shutdown()
            del self.cams[cam]

    async def status(self) -> List[str]:
        """Get status from cameras"""
        if self.dcam_initialized:
            for i, cam in enumerate(self):
                self._status[i] = STATUS[cam.get_status()]
            return self._status
        return [False, False]

    async def save_image(self, image_name: str, image_path: Union[str, Path]) -> int:
        """Save images to image_path/channel_image_name.tiff and return number of bytes saved."""
        nbytes = 0
        for cam in self:
            nbytes += cam.saveImage(image_name, image_path)

    async def set_exposure(self, time: float) -> float:
        """Sets the exposure time (s) for the camera.

        The definition for exposure changes between TDI and AREA mode:

        AREA mode:
        The camera alternates between exposure and readout according to the internal setting.
        Because the CCD is a full-frame transfer type, the readout time is contained in the exposure time.
        The exposure time can vary anywhere from 5.12 ms to 1004 ms in units of 20 μs.

        TDI mode:
        The exposure time is determined by the TDI line shift signal interval.
        The minimum interval that you can set is 20 μs.

        Args:
            time (float): The desired exposure time in seconds.

        Returns:
            float: The set exposure time in seconds.
        """
        for i, cam in enumerate(self):
            time = cam.setPropertyValue("exposure_time", time / 1000)
            self._exposure[i] = time * 1000
        return self._exposure

    async def get_exposure(self) -> float:
        """Retrieves the current exposure time (s) sfrom the camera.

        Returns:
            float: The current exposure time in seconds.
        """
        for i, cam in enumerate(self):
            self._exposure[i] = cam.getPropertyValue("exposure_time") * 1000
        return self._exposure

    @property
    def gain(self) -> List[int]:
        """Retrieves the cached contrast enhancement gain from the camera."""
        return self._gain

    async def set_gain(self, gain: int) -> int:
        """Sets the contrast enhancement gain (0 to 15) for the camera.

        Args:
            gain (int): The desired contrast enhancement gain
        """
        for i, cam in enumerate(self):
            self._gain[i] = cam.setPropertyValue("contrast_gain", gain)
        return gain

    async def get_gain(self):
        """Retrieves the current contrast enhancement gain from the camera.

        Returns:
            int: The current contrast enhancement gain.
        """
        for i, cam in enumerate(self):
            self._gain[i] = cam.getPropertyValue("contrast_gain")
        return self._gain

    async def capture(self):
        """Captures an image using the camera.

        This is an abstract asynchronous method that must be implemented by
        subclasses to define how the camera acquires an image. Captured images
        are assumed to be stored in the camera's internal memory.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    async def setTDI(self):
        """Put cameras into TDI mode."""
        for cam in self:
            cam.setTDI()

    async def setAREA(self):
        """Put cameras into AREA mode."""
        for cam in self:
            cam.setAREA()

    async def captureSetup(self):
        """Configure cameras for capturing new images"""
        for cam in self:
            cam.captureSetup()
