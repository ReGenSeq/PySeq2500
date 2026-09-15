from pyseq2500.com import COM_DICT
from pyseq2500.temperature import (
    ChillerTemperatureController,
    EmulatedARM9,
    FlowCellTemperatureController,
)
import pytest
import pytest_asyncio
import asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockARM9", marks=pytest.mark.mock),
        pytest.param("ARM9", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def arm9_com(request):
    """Single ARM9 COM shared across all three temperature controllers.

    In mock mode: one EmulatedARM9 instance.
    In hardware mode: the real SerialCOM from COM_DICT["ARM9"].
    """
    if request.param == "MockARM9":
        com = EmulatedARM9(address="ARM9COM")
    else:
        com = COM_DICT["ChillerTemperatureController"]
    await com.connect()
    yield com
    await com.close()


@pytest_asyncio.fixture(scope="class")
async def chiller(arm9_com):
    """ChillerTemperatureController — owns INIT/IDN, must initialize first."""
    tc = ChillerTemperatureController(
        name="ChillerTemperatureController",
        com=arm9_com,
    )
    yield tc


@pytest_asyncio.fixture(scope="class")
async def fc_tc_a(arm9_com):
    """FlowCellTemperatureController for channel A (fc_channel=0)."""
    tc = FlowCellTemperatureController(
        name="FlowCellTemperatureControllerA",
        com=arm9_com,
    )
    yield tc
    await tc.fc_off()


@pytest_asyncio.fixture(scope="class")
async def fc_tc_b(arm9_com):
    """FlowCellTemperatureController for channel B (fc_channel=1)."""
    tc = FlowCellTemperatureController(
        name="FlowCellTemperatureControllerB",
        com=arm9_com,
    )
    yield tc
    await tc.fc_off()


# ---------------------------------------------------------------------------
# ChillerTemperatureController
# ---------------------------------------------------------------------------


@pytest.mark.temperature
@pytest.mark.asyncio
class TestChillerTemperatureController:
    async def test_initialize(self, chiller: ChillerTemperatureController):
        """Chiller initializes the board (INIT + IDN) and sets PID params."""
        await chiller.initialize()

    async def test_set_get_temperature(self, chiller: ChillerTemperatureController):
        await chiller.set_temperature(4.0)
        temp = await chiller.get_temperature()
        assert abs(temp - 4.0) <= chiller.temperature_tolerance

    async def test_individual_tec_temperatures_populated(
        self, chiller: ChillerTemperatureController
    ):
        await chiller.get_temperature()
        assert len(chiller._chiller_temperatures) == 3
        assert all(t == t for t in chiller._chiller_temperatures)  # no NaN

    async def test_set_single_channel(self, chiller: ChillerTemperatureController):
        await chiller.set_temperature(6.0, channel=1)
        await chiller.get_temperature()
        assert (
            abs(chiller._chiller_temperatures[1] - 6.0) <= chiller.temperature_tolerance
        )

    async def test_wait_for_temperature(self, chiller: ChillerTemperatureController):
        await chiller.set_temperature(4.0)
        await asyncio.wait_for(chiller.wait_for_temperature(4.0), timeout=10)

    async def test_temperature_out_of_range_low(
        self, chiller: ChillerTemperatureController
    ):
        with pytest.raises(ValueError):
            await chiller.set_temperature(0.0)  # below min_val 0.1

    async def test_temperature_out_of_range_high(
        self, chiller: ChillerTemperatureController
    ):
        with pytest.raises(ValueError):
            await chiller.set_temperature(25.0)  # above max_val 20

    async def test_shutdown_is_noop(self, chiller: ChillerTemperatureController):
        """Chiller shutdown must not change state — reagents stay cold."""
        await chiller.set_temperature(4.0)
        await chiller.shutdown()
        temp = await chiller.get_temperature()
        assert abs(temp - 4.0) <= chiller.temperature_tolerance

    async def test_status(self, chiller: ChillerTemperatureController):
        assert await chiller.status() is True

    async def test_limits_from_config(self, chiller: ChillerTemperatureController):
        assert chiller.min_temperature == 0.1
        assert chiller.max_temperature == 20
        assert chiller.temperature_tolerance == 0.5


# ---------------------------------------------------------------------------
# FlowCellTemperatureController
# ---------------------------------------------------------------------------


@pytest.mark.temperature
@pytest.mark.asyncio
class TestFlowCellTemperatureController:
    async def test_initialize(
        self,
        chiller: ChillerTemperatureController,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        """Chiller must initialize before flowcell controllers."""
        await chiller.initialize()
        await asyncio.gather(fc_tc_a.initialize(), fc_tc_b.initialize())

    async def test_fc_channel_from_config(
        self,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        assert fc_tc_a.fc_channel == 0
        assert fc_tc_b.fc_channel == 1

    async def test_limits_from_config(self, fc_tc_a: FlowCellTemperatureController):
        assert fc_tc_a.min_temperature == 20
        assert fc_tc_a.max_temperature == 60
        assert fc_tc_a.temperature_tolerance == 0.5

    async def test_fc_on_off(self, fc_tc_a: FlowCellTemperatureController):
        await fc_tc_a.fc_on()
        await fc_tc_a.fc_off()

    async def test_set_get_temperature(self, fc_tc_a: FlowCellTemperatureController):
        await fc_tc_a.fc_on()
        await fc_tc_a.set_temperature(37.0)
        temp = await fc_tc_a.get_temperature()
        assert abs(temp - 37.0) <= fc_tc_a.temperature_tolerance

    async def test_wait_for_temperature(self, fc_tc_a: FlowCellTemperatureController):
        await fc_tc_a.set_temperature(37.0)
        await asyncio.wait_for(fc_tc_a.wait_for_temperature(37.0), timeout=10)

    async def test_temperature_out_of_range_low(
        self, fc_tc_a: FlowCellTemperatureController
    ):
        with pytest.raises(ValueError):
            await fc_tc_a.set_temperature(10.0)  # below min_val 20

    async def test_temperature_out_of_range_high(
        self, fc_tc_a: FlowCellTemperatureController
    ):
        with pytest.raises(ValueError):
            await fc_tc_a.set_temperature(70.0)  # above max_val 60

    async def test_channel_independence(
        self,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        """Setting channel A's temperature must not affect channel B."""
        await fc_tc_a.fc_on()
        await fc_tc_b.fc_on()
        await fc_tc_a.set_temperature(30.0)
        await fc_tc_b.set_temperature(40.0)
        temp_a = await fc_tc_a.get_temperature()
        temp_b = await fc_tc_b.get_temperature()
        assert abs(temp_a - 30.0) <= fc_tc_a.temperature_tolerance
        assert abs(temp_b - 40.0) <= fc_tc_b.temperature_tolerance

    async def test_concurrent_channels(
        self,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        """Both channels can be set concurrently through the shared COM lock."""
        await asyncio.gather(
            fc_tc_a.set_temperature(32.0),
            fc_tc_b.set_temperature(38.0),
        )
        temp_a = await fc_tc_a.get_temperature()
        temp_b = await fc_tc_b.get_temperature()
        assert abs(temp_a - 32.0) <= fc_tc_a.temperature_tolerance
        assert abs(temp_b - 38.0) <= fc_tc_b.temperature_tolerance

    async def test_status(self, fc_tc_a: FlowCellTemperatureController):
        await fc_tc_a.set_temperature(37.0)
        assert await fc_tc_a.status() is True

    async def test_shutdown(
        self,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        await asyncio.gather(fc_tc_a.shutdown(), fc_tc_b.shutdown())


# ---------------------------------------------------------------------------
# All three controllers concurrent
# ---------------------------------------------------------------------------


@pytest.mark.temperature
@pytest.mark.asyncio
class TestAllControllersConcurrent:
    async def test_all_three_concurrent(
        self,
        chiller: ChillerTemperatureController,
        fc_tc_a: FlowCellTemperatureController,
        fc_tc_b: FlowCellTemperatureController,
    ):
        """All three controllers can issue commands simultaneously through
        the shared COM lock without corrupting each other's channel state."""
        await asyncio.gather(
            fc_tc_a.set_temperature(35.0),
            fc_tc_b.set_temperature(37.0),
            chiller.set_temperature(4.0),
        )
        temp_a = await fc_tc_a.get_temperature()
        temp_b = await fc_tc_b.get_temperature()
        chiller_temp = await chiller.get_temperature()
        assert abs(temp_a - 35.0) <= fc_tc_a.temperature_tolerance
        assert abs(temp_b - 37.0) <= fc_tc_b.temperature_tolerance
        assert abs(chiller_temp - 4.0) <= chiller.temperature_tolerance
