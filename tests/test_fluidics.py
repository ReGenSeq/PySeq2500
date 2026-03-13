from pyseq2500.com import COM_DICT
from pyseq2500.fluidics import Pump, EmulatedPump, Valve, EmulatedValve
import pytest
import pytest_asyncio
import asyncio
from typing import List


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockPump", marks=pytest.mark.mock),
        pytest.param("Pump", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def pumps(request) -> List[Pump]:
    pumps = []
    for fc in ["A", "B"]:
        if request.param == "MockPump":
            com = EmulatedPump(name=f"Pump{fc}", address="PumpCOM")
        else:
            com = COM_DICT[f"Pump{fc}"]
        pumps.append(Pump(name=f"Pump{fc}", com=com, interval=0.5))

    yield pumps


async def wait_till_ready(pumps: List[Pump]):
    ready = [False, False]
    while not all(ready):
        for i, p in enumerate(pumps):
            if not p.ready:
                await asyncio.sleep(0.1)
            else:
                ready[i] = True


async def wait_till_busy(pumps: List[Pump]):
    busy = [False, False]
    while not all(busy):
        for i, p in enumerate(pumps):
            if p.ready:
                await asyncio.sleep(0.1)
            else:
                busy[i] = True


@pytest.mark.fluidic
@pytest.mark.asyncio
class TestPump:
    @pytest.mark.diagnostic
    async def test_init(self, pumps: List[Pump]):
        # Configure
        _ = []
        for p in pumps:
            _.append(p.configure())
        await asyncio.gather(*_)
        for p in pumps:
            assert p.min_volume < p.max_volume
            assert p.min_flow_rate < p.max_flow_rate
            assert p.barrels_per_lane > 0
            assert p.com.suffix == "\r"

        # Connect
        _ = []
        for p in pumps:
            _.append(p.connect())
        await asyncio.gather(*_)
        for p in pumps:
            assert p.connected

        # Initialize
        _ = []
        for p in pumps:
            _.append(p.initialize())
        await asyncio.gather(*_)
        for p in pumps:
            assert p.ready
            assert p.position == 0

    @pytest.mark.diagnostic
    async def test_pump(self, pumps: List[Pump]):
        p = pumps[0]
        vol = p.min_volume * 1000
        flow = p.min_flow_rate * 10
        out_flow = p.max_flow_rate * 0.8
        step = p.vol_to_step(vol)
        print("step", step)

        for p in pumps:
            asyncio.create_task(
                p.pump(vol, flow, pause_time=3, waste_flow_rate=out_flow)
            )

        # Wait for pump to finish aspiration
        await asyncio.sleep(vol / flow * 60)
        await wait_till_ready(pumps)
        for p in pumps:
            assert p.position == step

        # Wait for pump to start dispensing
        await wait_till_busy(pumps)

        # Wait for pump to finish dispensing
        await wait_till_ready(pumps)
        for p in pumps:
            assert p.position == 0

    ##        await asyncio.gather(*_)

    async def test_reverse_pump(self, pumps: List[Pump]):
        p = pumps[0]
        vol = p.min_volume * 1000
        flow = p.min_flow_rate * 10
        out_flow = p.max_flow_rate * 0.8
        step = p.vol_to_step(vol)

        _ = []
        for p in pumps:
            asyncio.create_task(
                p.reverse_pump(vol, out_flow, pause_time=3, waste_flow_rate=flow)
            )

        # Wait for pump to finish aspiration
        await asyncio.sleep(vol / flow * 60)
        await wait_till_ready(pumps)
        for p in pumps:
            assert p.position == step

        # Wait for pump to start dispensing
        await wait_till_busy(pumps)

        # Wait for pump to finish dispensing
        await wait_till_ready(pumps)
        for p in pumps:
            assert p.position == 0

    @pytest.mark.diagnostic
    async def test_shutdown(self, pumps: List[Pump]):
        _ = []
        for p in pumps:
            _.append(p.shutdown())
        await asyncio.gather(*_)
        for p in pumps:
            assert p.position == 0


@pytest.fixture(
    params=[
        pytest.param("MockValve", marks=pytest.mark.mock),
        pytest.param("Valve", marks=pytest.mark.hardware),
    ],
    scope="class",
)
def valves(request) -> List[Valve]:
    ##    name = request.param
    ##    if name == "MockValve":
    ##        _name = "Valve24B"
    ##        com = EmulatedValve(name=_name, address=request.param)
    ##    else:
    ##        com = COM_DICT[request.param]
    ##    yield Valve(name=com.name, com=com)

    valves = []
    for v in ["24A", "24B", "10A", "10B"]:
        name = f"Valve{v}"
        if request.param == "MockValve":
            com = EmulatedValve(name=name, address="ValveCOM")
        else:
            com = COM_DICT[name]
        valves.append(Valve(name=name, com=com))

    yield valves


@pytest.mark.fluidic
@pytest.mark.asyncio
@pytest.mark.diagnostic
class TestValve:
    async def test_init(self, valves: List[Valve]):
        _ = []
        for v in valves:
            _.append(v.com.connect())
        await asyncio.gather(*_)
        for v in valves:
            assert v.connected

        _ = []
        for v in valves:
            _.append(v.initialize())
        await asyncio.gather(*_)
        for v in valves:
            if "24" in v.name:
                assert v.n_ports == 24
            else:
                assert v.n_ports == 10
            assert v.port > 0

    async def test_select(self, valves: List[Valve]):
        _ = []
        for v in valves:
            port = v.config["port"]["valid_list"][1]
            _.append(v.select(port))
        await asyncio.gather(*_)
        for v in valves:
            port = v.config["port"]["valid_list"][1]
            assert v.port == port

        _ = []
        for v in valves:
            port = v.config["safe_port"]
            _.append(v.select(port))
        await asyncio.gather(*_)
        for v in valves:
            port = v.config["safe_port"]
            assert v.port == port

    async def test_shutdown(self, valves: List[Valve]):
        _ = []
        for v in valves:
            _.append(v.shutdown())
        await asyncio.gather(*_)
        for v in valves:
            port = v.config["safe_port"]
            assert v.port == port
