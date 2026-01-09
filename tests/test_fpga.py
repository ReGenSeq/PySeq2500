from pyseq2500.tiltstage import TiltStage, TiltMotor, EmulatedTiltMotor
from pyseq2500.fpga import EmulatedFPGA
from pyseq2500.led import LED, EmulatedLED
from pyseq2500.optics import FilterWheel, EmissionFilter, Shutter, EmulatedOptics
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockTiltStage", marks=pytest.mark.mock),
        pytest.param("TiltStage", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def tiltstage(request, fpga):
    if request.param == "MockTiltStage":
        com = EmulatedFPGA(address="FPGACOM")
        await com.connect()
        tiltstage = TiltStage(com=com)

        # Emulated coms for 3 motors
        coms = {}
        for i in range(1, 4):
            coms[i] = EmulatedTiltMotor(name=f"TiltMotor{i}", address="TiltStageCOM")
            await coms[i].connect()

        # Assign emulated coms to fake stage motors
        for id, com in coms.items():
            tiltstage.tilts[id] = TiltMotor(name=f"TiltMotor{id}", com=com)

    else:
        # Connect and initialize FPGA before running tests
        await fpga.connect()
        await fpga.initialize()
        tiltstage = TiltStage(com=fpga.com)

    yield tiltstage

    await tiltstage.shutdown()


@pytest.mark.stage
@pytest.mark.asyncio
class TestTiltStage:
    @pytest.mark.diagnostic
    async def test_init(self, tiltstage: TiltStage):
        await tiltstage.initialize()
        for pos in tiltstage.position:
            assert pos == 0

    @pytest.mark.diagnostic
    async def test_move(self, tiltstage: TiltStage):
        position = 5000
        await tiltstage.move(position)
        for i, pos in enumerate(tiltstage.position):
            assert abs(position - pos) <= tiltstage.tilts[i + 1].tolerance

    async def test_shutdown(self, tiltstage: TiltStage):
        await tiltstage.shutdown()
        assert await tiltstage.status()


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockFilterWheel", marks=pytest.mark.mock),
        pytest.param("GreenFilterWheel", marks=pytest.mark.hardware),
        pytest.param("RedFilterWheel", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def filterwheel(request, fpga):
    if request.param == "MockFilterWheel":
        com = EmulatedOptics(address="FPGACOM")
        await com.connect()
        filter = FilterWheel(name="GreenFilterWheel", com=com)
    else:
        # Connect and initialize FPGA before running tests
        await fpga.connect()
        await fpga.initialize()
        filter = FilterWheel(name=request.param, com=fpga.com)

    await filter.configure()
    yield filter

    await filter.shutdown()


@pytest.mark.optical
@pytest.mark.asyncio
class TestFilterWheel:
    @pytest.mark.diagnostic
    async def test_init(self, filterwheel: filterwheel):
        await filterwheel.initialize()
        assert filterwheel.filter == "home"

    @pytest.mark.diagnostic
    async def test_move(self, filterwheel: filterwheel):
        filter = "open"
        await filterwheel.set_filter(filter)
        assert filterwheel.filter == "open"

    async def test_shutdown(self, filterwheel: filterwheel):
        await filterwheel.shutdown()
        assert filterwheel.filter == "home"


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockEmissionFilter", marks=pytest.mark.mock),
        pytest.param("EmissionFilter", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def emissionfilter(request, fpga):
    if request.param == "MockEmissionFilter":
        com = EmulatedOptics(address="FPGACOM")
        await com.connect()
        filter = EmissionFilter(com=com)
    else:
        # Connect and initialize FPGA before running tests
        await fpga.connect()
        await fpga.initialize()
        filter = EmissionFilter(name=request.param, com=fpga.com)

    await filter.configure()
    yield filter

    await filter.shutdown()


@pytest.mark.optical
@pytest.mark.asyncio
class TestEmissionFilter:
    @pytest.mark.diagnostic
    async def test_init(self, emissionfilter: emissionfilter):
        await emissionfilter.initialize()
        assert emissionfilter.filter == "in"

    @pytest.mark.diagnostic
    async def test_move(self, emissionfilter: emissionfilter):
        filter = "out"
        await emissionfilter.set_filter(filter)
        assert emissionfilter.filter == filter

    async def test_shutdown(self, emissionfilter: emissionfilter):
        await emissionfilter.shutdown()
        assert emissionfilter.filter == "in"


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockShutter", marks=pytest.mark.mock),
        pytest.param("Shutter", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def shutter(request, fpga):
    if request.param == "MockShutter":
        com = EmulatedOptics(address="FPGACOM")
        await com.connect()
        shutter = Shutter(com=com)
    else:
        # Connect and initialize FPGA before running tests
        await fpga.connect()
        await fpga.initialize()
        shutter = Shutter(name=request.param, com=fpga.com)

    yield shutter

    await shutter.shutdown()


@pytest.mark.optical
@pytest.mark.asyncio
class TestShutter:
    @pytest.mark.diagnostic
    async def test_init(self, shutter: shutter):
        await shutter.initialize()
        assert not shutter.is_open

    @pytest.mark.diagnostic
    async def test_open(self, shutter: shutter):
        await shutter.open()
        assert shutter.is_open

    async def test_shutdown(self, shutter: shutter):
        await shutter.shutdown()
        assert not shutter.is_open


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockLEDA", marks=pytest.mark.mock),
        pytest.param("LEDA", marks=pytest.mark.hardware),
        pytest.param("LEDB", marks=pytest.mark.hardware),
    ],
    scope="session",
)
async def led(request, fpga):
    if request.param == "MockLEDA":
        com = EmulatedLED(address="FPGACOM")
        await com.connect()
        led = LED(name=request.param, com=com)
    else:
        # Connect and initialize FPGA before running tests
        await fpga.connect()
        await fpga.initialize()
        led = LED(name=request.param, com=fpga.com)

    await led.configure()

    yield led

    await led.shutdown()


@pytest.mark.optical
@pytest.mark.asyncio
class TestLED:
    @pytest.mark.diagnostic
    async def test_init(self, led: led):
        await led.initialize()
        assert led.status

    @pytest.mark.diagnostic
    async def test_open(self, led: led):
        await led.set_led("imaging")
        assert led.status

    async def test_shutdown(self, led: led):
        await led.shutdown()
        assert led.status
