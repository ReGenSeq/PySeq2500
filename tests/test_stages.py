from pyseq2500.com import COM_DICT
from pyseq2500.xstage import XStage, EmulatedXStage
from pyseq2500.ystage import EmulatedYStage, YStage
from pyseq2500.tiltstage import TiltStage, TiltMotor, EmulatedTiltMotor
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockXStage", marks=pytest.mark.mock),
        pytest.param("XStage", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def xstage(request):
    name = request.param
    if name == "MockXStage":
        com = EmulatedXStage(address="XStageCOM")
    else:
        com = COM_DICT[request.param]

    xstage = XStage(com=com)

    # Connect to COM
    await xstage.com.connect()
    assert xstage.connected

    yield xstage

    await xstage.shutdown()


@pytest.mark.stage
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestXStage:
    async def test_init(self, xstage: XStage):
        await xstage.initialize()
        assert xstage.position == xstage.home_position

    async def test_move(self, xstage: XStage):
        await xstage.move(31000)
        assert xstage.position == 31000

    async def test_shutdown(self, xstage: XStage):
        await xstage.shutdown()
        assert xstage.position == xstage.home_position


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockYStage", marks=pytest.mark.mock),
        pytest.param("YStage", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def ystage(request):
    name = request.param
    if name == "MockYStage":
        com = EmulatedYStage(address="YStageCOM")
    else:
        com = COM_DICT[request.param]

    ystage = YStage(com=com)

    # Connect to COM
    await ystage.com.connect()
    assert ystage.connected

    yield ystage

    await ystage.shutdown()


@pytest.mark.stage
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestYStage:
    # Don't test against precise stage position
    # precision ~< 5 steps
    # Test status which returns in position flag as true
    async def test_init(self, ystage: YStage):
        await ystage.initialize()
        assert await ystage.status()

    async def test_move(self, ystage: YStage):
        await ystage.move(1000000)
        assert await ystage.status()

    async def test_shutdown(self, ystage: YStage):
        await ystage.shutdown()
        assert await ystage.status()


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockTiltStage", marks=pytest.mark.mock),
        pytest.param("TiltStage", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def tiltstage(request):
    name = request.param
    if name == "MockTiltStage":
        # Fake com not used to create TiltStage
        dumcom = EmulatedTiltMotor(name="TiltStage", address="TiltStageCOM")
        tiltstage = TiltStage(com=dumcom)

        # Emulated coms for 3 motors
        coms = {}
        for i in range(1, 4):
            coms[i] = EmulatedTiltMotor(name=f"TiltMotor{i}", address="TiltStageCOM")
            await coms[i].connect()

        # Assign emulated coms to fake stage motors
        for id, com in coms.items():
            tiltstage.tilts[id] = TiltMotor(name=f"TiltMotor{id}", com=com)

    else:
        com = COM_DICT[request.param]
        tiltstage = TiltStage(com=com)

    # Connect to COM
    await tiltstage.com.connect()
    assert tiltstage.com.connected

    yield tiltstage

    await tiltstage.shutdown()


@pytest.mark.stage
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestTiltStage:
    async def test_init(self, tiltstage: TiltStage):
        await tiltstage.initialize()
        for pos in tiltstage.position:
            assert pos == 0

    async def test_move(self, tiltstage: TiltStage):
        print(tiltstage.tilts[1].config)
        position = 5000
        await tiltstage.move(position)
        for i, pos in enumerate(tiltstage.position):
            assert abs(position - pos) <= tiltstage.tilts[i + 1].tolerance

    async def test_shutdown(self, tiltstage: TiltStage):
        await tiltstage.shutdown()
        assert await tiltstage.status()
