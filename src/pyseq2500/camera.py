import logging
from pyseq_core.base_instruments import BaseCamera
from pyseq2500.utils import HW_CONFIG
from attrs import define, field
from typing import Union, List, Literal
from pathlib import Path
import ctypes
from pyseq_core.dcam import DCAMException
import asyncio
import numpy as np

LOGGER = logging.getLogger("PySeq")

STATUS = {0: "error", 1: "busy", 2: "ready", 3: "stable", 4: "unstable"}


@define
class EmulatedTDICamera:
    camera_id: int = field()
    exposure: float = field(default=1.0)
    gain: float = field(default=1.0)
    status: int = field(default=3)
    properties: dict = field(factory=dict)
    number_image_buffers: int = field(default=0)
    left_emission: str = field(init=False)
    right_emission: str = field(init=False)
    sensor_mode: Literal["TDI", "AREA"] = field(default="TDI")

    def saveImage(self, image_name, image_path):
        LOGGER.debug(f"Saving {image_path}/c{self.left_emission}_{image_name}.tiff")
        LOGGER.debug(f"Saving {image_path}/c{self.right_emission}_{image_name}.tiff")
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
        self.setPropertyValue("sensor_mode_line_bundle_height", 64)
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

    def getFrameInterval(self) -> float:
        return 0.040202

    def getFocusStack(self) -> np.ndarray:
        # Return array of arrays (image data per frame per channel)
        # Shape: (n_frames, 2) with dtype=object for this camera's 2 channels
        n_frames = self.number_image_buffers
        focus_stack = np.empty((n_frames, 2), dtype=object)
        for i in range(n_frames):
            for j in range(2):
                focus_stack[i, j] = (
                    np.ones((64, HW_CONFIG["Cameras"]["sensor_width"]), dtype=np.uint16)
                    * 1000
                )
        return focus_stack


@define
class dcamCOM:
    _connected: bool = field(default=False)
    cams: dict[int, EmulatedTDICamera] = field(factory=dict)
    emulated: bool = field(default=False)

    @property
    def connected(self):
        return self._connected

    async def connect(self):
        if not self.emulated:
            try:
                self.emulated = False
                from pyseq_core.dcam import HamamatsuCamera

                dcam = ctypes.windll.dcamapi
                temp = ctypes.c_int32(0)
                if dcam.dcam_init(None, ctypes.byref(temp), None) != 1:
                    raise DCAMException("DCAM initialization failed.")
                n_cameras = temp.value
                LOGGER.debug(f"DCAM found {n_cameras} cameras")
            except AttributeError as e:
                LOGGER.error(e)
                LOGGER.warning("DCAM is not installed, okay for testing or no imaging")
                self.emulated = True
            except DCAMException as e:
                LOGGER.error(e)
                LOGGER.error("DCAM failed, restart system")
                self.emulated = True

        try:
            for i in range(2):
                if i not in self.cams:
                    if self.emulated:
                        self.cams[i] = EmulatedTDICamera(i)
                        LOGGER.debug(f"Camera {i} connected to CAM{i}")
                    else:
                        self.cams[i] = HamamatsuCamera(i)
                        LOGGER.debug(f"Camera {i} connected to dcam {i}")
            self._connected = True
        except DCAMException as e:
            LOGGER.error(e)
            LOGGER.error(f"DCAM could not connect to Camera {i}")


@define(kw_only=True)
class TDICameras(BaseCamera):
    """Wrapper around 2 instances of pyseq_core.dcam.HammatsuCamera."""

    name: str = field(default="Cameras")
    com: dcamCOM = field(default=dcamCOM())  # pyright: ignore[reportIncompatibleVariableOverride]
    _status: list = field(default=[None, None])
    _exposure: list = field(default=[None, None])  # pyright: ignore[reportIncompatibleVariableOverride]
    _gain: list = field(default=[None, None])
    mode: Literal["TDI", "AREA"] = field(default="TDI")

    @property
    def cams(self) -> dict[int, EmulatedTDICamera]:
        return self.com.cams

    def __iter__(self):
        for cam in self.com.cams.values():
            yield cam

    async def initialize(self):
        """Initialize cameras to TDI mode."""
        for cam in self:
            cam.setTDI()
            cam.captureSetup()

        await self.status()
        await self.get_exposure()
        await self.get_gain()

    async def configure(self, exp_config: dict = {}):
        """Configure channel names."""

        for cam in self.cams.values():
            LOGGER.debug(f"configuring camera {cam.camera_id}")
            if exp_config.get("cameras", {}).get("save_nt", False):
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
        self.com._connected = False

    async def status(self) -> bool:
        """Get status from cameras"""
        if self.com._connected:
            for i, cam in enumerate(self):
                self._status[i] = STATUS[cam.get_status()]
            return all(self._status)
        return False

    async def save_image(self, image_name: str, image_dir: Union[str, Path]) -> int:
        """Save images to image_dir/channel_image_name.tiff and return number of bytes saved."""
        nbytes = 0
        for cam in self:
            nbytes += cam.saveImage(image_name, image_dir)
        return nbytes

    async def set_exposure(
        self, exposure: Union[float, dict[str, float]]
    ) -> List[float]:
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
            exposure (float | dict[str, float]): The desired exposure time in seconds,
                or a dict mapping camera names to exposure times.

        Returns:
            List[float]: The set exposure times in seconds.
        """
        if isinstance(exposure, dict):
            for cam in self.cams.values():
                cam_name = self.config[cam.camera_id]["name"]
                if cam_name in exposure:
                    time = exposure[cam_name]
                    cam.setPropertyValue("exposure_time", time / 1000)
                    time = cam.getPropertyValue("exposure_time")
                    self._exposure[cam.camera_id] = time * 1000
        else:
            for i, cam in self.cams.items():
                cam.setPropertyValue("exposure_time", exposure / 1000)
                time = cam.getPropertyValue("exposure_time")
                self._exposure[i] = time * 1000
        return self._exposure

    async def get_exposure(self) -> List[float]:
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

    async def check_free(self):
        await self.status()
        while not all(self._status):
            for cam_id, status in enumerate(self._status):
                if not status:
                    self.cams[cam_id].stopAcquisition()
                    self.cams[cam_id].freeFrames()
                    self.cams[cam_id].captureSetup()
                    self._status[cam_id] = self.cams[cam_id].get_status()

    async def allocate(self, mode: Literal["TDI", "AREA"], n_frames: int):
        # Set mode
        if mode == "TDI":
            await self.setTDI()
        elif mode == "AREA":
            await self.setAREA()
        else:
            raise ValueError(f"Unknown mode {mode}. Must be TDI or AREA")

        # Allocate memory for image data
        for cam in self:
            cam.allocFrame(n_frames)

    async def setTDI(self):
        """Put cameras into TDI mode."""
        if self.mode != "TDI":
            for cam in self:
                cam.setTDI()
                cam.captureSetup()
            self.mode = "TDI"

    async def setAREA(self):
        """Put cameras into AREA mode."""
        if self.mode != "AREA":
            for cam in self:
                cam.setAREA()
                cam.captureSetup()
            self.mode = "AREA"

    async def captureSetup(self):
        """Configure cameras for capturing new images"""
        for cam in self:
            cam.captureSetup()

    async def startAcquisition(self):
        for cam in self:
            cam.startAcquisition()

    async def stopAcquisition(self):
        for cam in self:
            cam.stopAcquisition()

    async def freeFrames(self):
        for cam in self:
            cam.freeFrames()

    async def getFrameCount(self):
        nframes = [0, 0]
        for cam in self:
            camid = cam.camera_id
            nframes[camid] += cam.getFrameCount()
            LOGGER.debug(f"Camera{camid}:: Took {nframes[camid]} frames")
        return (nframes[0] + nframes[1]) / 2

    async def waitForFrames(self, n_frames: int):
        est_time = 0
        for cam in self:
            interval = cam.getFrameInterval()
            est_time += interval * n_frames

        async with asyncio.timeout(est_time):
            while await self.getFrameCount() < n_frames:
                await asyncio.sleep(0.5)

    async def getFocusStack(self):
        stack = []
        for cam in self:
            stack.append(cam.getFocusStack())
        return np.hstack(stack)
