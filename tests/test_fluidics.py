from pyseq2500.com import COM_DICT
from pyseq2500.fluidics import Pump, EmulatedPump, Valve, EmulatedValve
import pytest
import asyncio


@pytest.fixture(
    params=[
        pytest.param("MockPump", marks=pytest.mark.mock),
        pytest.param("PumpA", marks=pytest.mark.hardware),
        pytest.param("PumpB", marks=pytest.mark.hardware),
    ],
    scope="class",
)
def pump(request) -> Pump:
    name = request.param
    if name == "MockPump":
        com = EmulatedPump(name="PumpB", address="PumpCOM")
        interval = 0.5
    else:
        com = COM_DICT[request.param]
        interval = 1
    return Pump(name=com.name, com=com, interval=interval)


@pytest.mark.fluidic
@pytest.mark.asyncio
class TestPump:
    @pytest.mark.diagnostic
    async def test_init(self, pump: Pump):
        await pump.configure()
        assert pump.min_volume < pump.max_volume
        assert pump.min_flow_rate < pump.max_flow_rate
        assert pump.barrels_per_lane > 0

        await pump.com.connect()
        assert pump.connected

        await pump.initialize()
        assert pump.ready
        assert pump.position == 0

    @pytest.mark.diagnostic
    async def test_pump(self, pump: Pump):
        vol = pump.min_volume * 1000
        flow = pump.min_flow_rate * 10
        out_flow = pump.max_flow_rate * 0.8
        asyncio.create_task(
            pump.pump(vol, flow, pause_time=3, waste_flow_rate=out_flow)
        )
        step = pump.vol_to_step(vol)

        # Wait for pump to finish aspiratin
        await asyncio.sleep(vol / flow * 60)
        while not pump.ready:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)

        assert pump.position == step

        # Wait for pump to start dispensing
        while pump.ready:
            await asyncio.sleep(0.1)
        # Wait for pump to finish dispensing
        while not pump.ready:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)

        assert pump.position == 0

    async def test_reverse_pump(self, pump: Pump):
        vol = pump.min_volume * 1000
        out_flow = pump.min_flow_rate * 10
        in_flow = pump.max_flow_rate * 0.5
        asyncio.create_task(
            pump.reverse_pump(vol, out_flow, pause_time=3, waste_flow_rate=in_flow)
        )
        step = pump.vol_to_step(vol)

        # Wait for pump to finish aspiratin
        await asyncio.sleep(vol / in_flow * 60)
        while not pump.ready:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)

        assert pump.position == step

        # Wait for pump to start dispensing
        await asyncio.sleep(vol / out_flow * 60)
        while pump.ready:
            await asyncio.sleep(0.1)
        # Wait for pump to finish dispensing
        while not pump.ready:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)

        assert pump.position == 0

    @pytest.mark.diagnostic
    async def test_shutdown(self, pump: Pump):
        await pump.shutdown()
        assert pump.position == 0


@pytest.fixture(
    params=[
        pytest.param("MockValve", marks=pytest.mark.mock),
        pytest.param("Valve24A", marks=pytest.mark.hardware),
        pytest.param("Valve24B", marks=pytest.mark.hardware),
        pytest.param("Valve10A", marks=pytest.mark.hardware),
        pytest.param("Valve10B", marks=pytest.mark.hardware),
    ],
    scope="class",
)
def valve(request) -> Valve:
    name = request.param
    if name == "MockValve":
        _name = "Valve24B"
        com = EmulatedValve(name=_name, address=request.param)
    else:
        com = COM_DICT[request.param]
    return Valve(name=com.name, com=com)


@pytest.mark.fluidic
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestValve:
    async def test_init(self, valve: Valve):
        await valve.com.connect()
        assert valve.com.com is not None

        await valve.initialize()
        assert valve.n_ports == 10 or valve.n_ports == 24
        assert valve.port > 0

    async def test_select(self, valve: Valve):
        port = valve.config["port"]["valid_list"][1]
        await valve.select(port)
        assert valve.port == port

        port = valve.config["safe_port"]
        await valve.select(port)
        assert valve.port == port

    async def test_shutdown(self, valve: Valve):
        await valve.shutdown()
        assert valve.port == valve.config["safe_port"]
