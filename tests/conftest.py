import pytest_asyncio
import pytest
from pyseq2500.sequencer import PySeq2500
from os import mkdir
import asyncio
import tarfile

import pooch
import numpy as np
import imageio.v2 as imageio

from pyseq2500.microscope import Microscope
from pyseq2500.xstage import EmulatedXStage, XStage
from pyseq2500.ystage import EmulatedYStage, YStage
from pyseq2500.zstage import EmulatedZStage, ZStage
from pyseq2500.tiltstage import EmulatedTiltMotor, TiltStage, TiltMotor
from pyseq2500.fpga import EmulatedFPGA, FPGA
from pyseq2500.laser import EmulatedLaser, Laser
from pyseq2500.optics import FilterWheel, EmissionFilter, Shutter, EmulatedOptics
from pyseq2500.camera import TDICameras, dcamCOM
from pyseq_core.base_protocol import BaseROI


# Base Test FPGA
@pytest_asyncio.fixture(scope="session")
async def fpga():
    """FPGA for tilt stage, obj stage, filters, shutter and LEDs."""

    from pyseq2500.fpga import EmulatedFPGA

    fpga = EmulatedFPGA(address="FPGACOM")

    yield fpga


@pytest.fixture(scope="session")
def rough_scan_data():
    """Download and cache RoughScan test data from Zenodo.

    Returns:
        Path: Path to the extracted RoughScan directory containing image files.
              The directory contains:
              - Image files: 12-bit TIFF files across 4 channels (558, 610, 687, 740)
                with 3 tiles per channel (tile IDs: 10292, 10607, 10922)
              - Metadata files: .txt files with scan parameters
    """
    # Use pooch's default cache directory (~/.cache/pyseq2500)
    data_path = pooch.os_cache("pyseq2500")

    # Create a pooch instance for the Zenodo dataset
    rough_pooch = pooch.create(
        path=data_path,
        base_url="https://zenodo.org/record/19355899/files/",
        registry={"RoughScan.tar.gz": None},  # None skips checksum for now
    )

    # Download the tarball (cached after first download)
    tarball_path = rough_pooch.fetch("RoughScan.tar.gz", processor=None)

    # Extract the tarball
    extract_dir = data_path / "RoughScan"
    if not extract_dir.exists():
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=data_path)

        # Clean up macOS resource files (._*)
        for f in extract_dir.glob("._*"):
            f.unlink(missing_ok=True)

    return extract_dir


@pytest.fixture(scope="session")
def focus_stack_data():
    """Download and cache focus stack test data from Zenodo.

    Returns:
        dict: Dictionary with camera IDs as keys and focus stack arrays as values.
              Each focus stack is shape (n_frames, 2) with dtype=object,
              where each element is a 16x2048 uint16 numpy array.
              Camera 0: channels 687, 558
              Camera 1: channels 610, 740
    """

    # Use pooch's default cache directory (~/.cache/pyseq2500)
    data_path = pooch.os_cache("pyseq2500")

    # Create a pooch instance for the Zenodo dataset
    focus_pooch = pooch.create(
        path=data_path,
        base_url="https://zenodo.org/record/19355899/files/",
        registry={"focus_stack.tar.gz": "md5:b822df351a597531674801f356ca39c0"},
    )

    # Download the tarball (cached after first download)
    tarball_path = focus_pooch.fetch("focus_stack.tar.gz", processor=None)

    # Extract the tarball
    extract_dir = data_path / "focus_stack"
    if not extract_dir.exists():
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=data_path)

    # Load focus stack data
    # The focus stack contains tiff files for 4 channels (558, 610, 687, 740)
    # Each channel has multiple Z-frames
    # Camera 0: 687 (left), 558 (right)
    # Camera 1: 610 (left), 740 (right)
    camera_channels = {
        0: [687, 558],
        1: [610, 740],
    }

    channel_data = {ch: [] for ch in [687, 558, 610, 740]}

    # Find and load all tiff files, sorted by channel and frame
    tiff_files = sorted(extract_dir.glob("*.tiff"))

    for tiff_file in tiff_files:
        # Extract channel from filename (e.g., c558_f0.tiff)
        filename = tiff_file.name
        for ch in channel_data.keys():
            if f"c{ch}_" in filename:
                img = imageio.imread(str(tiff_file))
                channel_data[ch].append(img)
                break

    # Determine number of frames (assuming all channels have same number)
    n_frames = len(channel_data[687])

    # Create focus stack arrays per camera: (n_frames, 2) with dtype=object
    # Each element is a 16x2048 uint16 array
    camera_stacks = {}
    for cam_id, channels in camera_channels.items():
        cam_stack = np.empty((n_frames, 2), dtype=object)
        for i in range(n_frames):
            for j, ch in enumerate(channels):
                cam_stack[i, j] = channel_data[ch][i]
        camera_stacks[cam_id] = cam_stack

    return camera_stacks


@pytest.fixture(scope="session")
def rough_scan_hiseq(rough_scan_data):
    """Load RoughScan data using HiSeqImages.open_RoughScan.

    Returns:
        HiSeqImages: Loaded and processed HiSeqImages object with:
            - Images opened with open_RoughScan(suffix='tif')
            - Background correction applied
            - Channel registration applied
            - im attribute is xarray.DataArray with shape (4, ~4452, ~6143)
              channels: [558, 610, 687, 740]
    """
    from pre import image_analysis as ia

    hiseq = ia.HiSeqImages.open_RoughScan(str(rough_scan_data), [".tiff", ".tif"])
    hiseq.correct_background()
    hiseq.register_channels()

    return hiseq


@pytest_asyncio.fixture(scope="session")
def test_directory(tmp_path_factory):
    """Directory for tests."""
    exp_path = tmp_path_factory.mktemp("pyseq2500test")
    mkdir(exp_path / "images")
    mkdir(exp_path / "focus")
    mkdir(exp_path / "logs")

    return exp_path


@pytest_asyncio.fixture(scope="session")
async def fc_A_roi(test_directory):
    """Test region of interest to image on flow cell A."""

    from pyseq_core.base_protocol import CUSTOM_ROI

    image_fields = {"nz": 1, "image_dir": str(test_directory / "images")}
    focus_fields = {"output": str(test_directory / "focus")}
    roi = CUSTOM_ROI(
        flowcell="A", LLx=17.462, LLy=35.5, URx=15.768, URy=34.252, overlap=0
    )
    stage = PySeq2500.custom_roi_stage(roi)

    return BaseROI(name="centerA", stage=stage, image=image_fields, focus=focus_fields)


@pytest_asyncio.fixture(scope="session")
async def fc_B_roi(test_directory):
    """Test region of interest to image on flow cell B."""

    from pyseq_core.base_protocol import CUSTOM_ROI

    image_fields = {"nz": 1, "image_dir": str(test_directory / "images")}
    focus_fields = {"output": str(test_directory / "focus")}
    roi = CUSTOM_ROI(
        flowcell="B", LLx=12.75, LLy=37.75, URx=12.25, URy=37.25, overlap=0
    )
    stage = PySeq2500.custom_roi_stage(roi)

    return BaseROI(name="centerB", stage=stage, image=image_fields, focus=focus_fields)


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockMicroscope", marks=pytest.mark.mock),
        pytest.param("Microscope", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def microscope(request, fpga, focus_stack_data):
    name = request.param
    # Construct Microscope
    m = Microscope()
    if name == "MockMicroscope":
        fcom = EmulatedFPGA(address="FPGACOM")
        xcom = EmulatedXStage(name="XStage", address="XStageCOM")
        ycom = EmulatedYStage(name="YStage", address="YStageCOM")
        zcom = EmulatedZStage(name="ZStage", address="ZStageCOM")
        tcom = EmulatedTiltMotor(name="TiltStage", address="FPGA")
        glcom = EmulatedLaser(name="GreenLaser", address="LaserCOM")
        rlcom = EmulatedLaser(name="RedLaser", address="LaserCOM")
        gfwcom = EmulatedOptics(name="GreenFilterWheel", address="FPGA")
        rfwcom = EmulatedOptics(name="RedFilterWheel", address="FPGA")
        ecom = EmulatedOptics(name="EmissionFilter", address="FPGA")
        scom = EmulatedOptics(name="Shutter", address="FPGA")
        dcam = dcamCOM(emulated=True, focus_stack=focus_stack_data)

        instruments = {
            "FPGA": FPGA(com=fcom),
            "XStage": XStage(name="XStage", com=xcom),
            "YStage": YStage(name="YStage", com=ycom),
            "ZStage": ZStage(name="ZStage", com=zcom),
            "TiltStage": TiltStage(name="TiltStage", com=tcom),
            "Lasers": {
                "green": Laser(name="GreenLaser", com=glcom, color="green"),
                "red": Laser(name="RedLaser", com=rlcom, color="red"),
            },
            "FilterWheels": {
                "green": FilterWheel(name="GreenFilterWheel", com=gfwcom),
                "red": FilterWheel(name="RedFilterWheel", com=rfwcom),
            },
            "EmissionFilter": EmissionFilter(name="EmissionFilter", com=ecom),
            "Shutter": Shutter(name="Shutter", com=scom),
            "Camera": TDICameras(com=dcam),
        }

        # Emulated coms for 3 motors
        coms = {}
        for i in range(1, 4):
            coms[i] = EmulatedTiltMotor(name=f"TiltMotor{i}", address="FPGA")
            await coms[i].connect()
        # Assign emulated coms to fake stage motors
        for id, com in coms.items():
            instruments["TiltStage"].tilts[id] = TiltMotor(
                name=f"TiltMotor{id}", com=com
            )

        m.instruments = instruments
        m.name = "MockMicroscope"
    else:
        # Attach fpga com fixture
        fpga_instruments = [
            "FPGA",
            "ZStage",
            "TiltStage",
            "EmissionFilter",
            "Shutter",
        ]
        for fi in fpga_instruments:
            m.instruments[fi].com = fpga
        for fi in m.instruments["FilterWheels"].values():
            fi.com = fpga

    # Start async worker
    m.start()

    await m._connect()
    await m._configure({})

    # Do tests
    yield m

    # Shutdown instruments
    await m._shutdown()

    # Shutdown async worker gracefully
    try:
        m._worker_task.cancel()
    except asyncio.CancelledError:
        await m._worker_task
