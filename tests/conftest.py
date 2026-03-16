import pytest_asyncio
from pyseq2500.com import COM_DICT
from pyseq2500.utils import DEFAULT_CONFIG
from pyseq2500.fpga import FPGA
from pyseq2500.sequencer import PySeq2500
from pyseq_core.base_protocol import ROIFactory
from os import mkdir


# Base Test FPGA
@pytest_asyncio.fixture(scope="session")
async def fpga():
    """FPGA for tilt stage, obj stage, filters, shutter and LEDs."""

    fpga = FPGA(com=COM_DICT["FPGA"])

    yield fpga


@pytest_asyncio.fixture(scope="session")
def test_directory(tmp_path_factory):
    """Directory for tests."""
    exp_path = tmp_path_factory.mktemp("pyseq2500test")
    mkdir(exp_path / "images")
    mkdir(exp_path / "focus")
    mkdir(exp_path / "logs")

    return exp_path


@pytest_asyncio.fixture(scope="function")
async def fc_A_roi(test_directory):
    """Test region of interest to image on flow cell A."""

    test_conf = DEFAULT_CONFIG.copy()
    test_conf["experiment"]["images_path"] = str(test_directory / "images")
    test_conf["image"]["nz"] = 1
    TestROI = ROIFactory.factory(test_conf)

    stage = PySeq2500.custom_roi_stage("A", LLx=12.75, LLy=37.75, URx=12.25, URy=37.25)

    return TestROI(name="centerA", stage=stage)


@pytest_asyncio.fixture(scope="function")
async def fc_B_roi(test_directory):
    """Test region of interest to image on flow cell B."""

    test_conf = DEFAULT_CONFIG.copy()
    test_conf["experiment"]["images_path"] = str(test_directory / "images")
    test_conf["image"]["nz"] = 1
    TestROI = ROIFactory.factory(test_conf)

    stage = PySeq2500.custom_roi_stage("B", LLx=12.75, LLy=37.75, URx=12.25, URy=37.25)

    return TestROI(name="centerB", stage=stage)
