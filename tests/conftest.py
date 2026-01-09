import pytest_asyncio
from pyseq2500.com import COM_DICT
from pyseq2500.fpga import FPGA


# Base Test Sequencer
@pytest_asyncio.fixture(scope="session")
async def fpga():
    """FPGA for tilt stage, obj stage, filters, shutter and LEDs."""

    fpga = FPGA(com=COM_DICT["FPGA"])

    yield fpga
