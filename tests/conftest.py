import pytest_asyncio
from pyseq2500.com import COM_DICT
from pyseq2500.utils import DEFAULT_CONFIG
from pyseq2500.fpga import FPGA
from pyseq2500.sequencer import PySeq2500
from pyseq_core.base_protocol import ROIFactory

DefaultROI = ROIFactory.factory(DEFAULT_CONFIG)


# Base Test Sequencer
@pytest_asyncio.fixture(scope="session")
async def fpga():
    """FPGA for tilt stage, obj stage, filters, shutter and LEDs."""

    fpga = FPGA(com=COM_DICT["FPGA"])

    yield fpga


@pytest_asyncio.fixture(scope="function")
async def fc_A_roi():
    """Test region of interest to image on flow cell A."""

    stage = PySeq2500.custom_roi_stage("A", LLx=12.75, LLy=37.75, URx=12.25, URy=37.25)
    stage["nz"] = 1

    return DefaultROI(name="centerA", stage=stage)


@pytest_asyncio.fixture(scope="function")
async def fc_B_roi():
    """Test region of interest to image on flow cell B."""

    stage = PySeq2500.custom_roi_stage("B", LLx=12.75, LLy=37.75, URx=12.25, URy=37.25)
    stage["nz"] = 1

    return DefaultROI(name="centerB", stage=stage)
