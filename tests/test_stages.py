from pyseq2500.com import COM_DICT
from pyseq2500.stages import XStage, EmulatedXStage
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockXStage", marks=pytest.mark.mock),
        pytest.param("XStage", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def xstage(request) -> XStage:
    name = request.param
    if name == "MockXStage":
        com = EmulatedXStage(address="XStageCOM")
    else:
        com = COM_DICT[request.param]

    xstage = XStage(com=com)

    # Connect to COM
    await xstage.com.connect()
    assert xstage.connected

    return xstage


@pytest.mark.stage
@pytest.mark.asyncio
class TestXStage:
    @pytest.mark.diagnostic
    async def test_init(self, xstage: XStage):
        await xstage.initialize()

    @pytest.mark.diagnostic
    async def test_move(self, xstage: XStage):
        await xstage.move(31000)
        assert xstage.position == 31000

    @pytest.mark.diagnostic
    async def test_shutdown(self, xstage: XStage):
        await xstage.shutdown()
        assert xstage.position == xstage.home_position
