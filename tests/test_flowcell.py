from pyseq2500.fluidics import Pump, Valve, EmulatedPump, EmulatedValve
from pyseq2500.flowcell import FlowCell
from pyseq2500.utils import DEFAULT_CONFIG
import pytest
import pytest_asyncio
import asyncio


@pytest_asyncio.fixture(
    params=[
        pytest.param("B", marks=pytest.mark.mock),
        pytest.param("FlowCellA", marks=pytest.mark.hardware),
        pytest.param("FlowCellB", marks=pytest.mark.hardware),
    ],
    scope="class",
)
async def fc(request) -> FlowCell:
    name = request.param
    # Construct FlowCell
    if name == "B":
        pcom = EmulatedPump(name="PumpB", address="PumpBCOM")
        v24com = EmulatedValve(name="Valve24B", address="Valve24BCOM")
        v10com = EmulatedValve(name="Valve10B", address="Valve10BCOM")
        instruments = {
            "Pump": Pump(name="PumpB", com=pcom),
            "Valve": Valve(name="Valve24B", com=v24com),
            "InletValve": Valve(name="Valve10B", com=v10com),
        }
        fc = FlowCell(name="FlowCellB")
        fc.instruments = instruments
    else:
        fc = FlowCell(name=name)
    # Start async worker
    fc.start()

    # Do tests
    yield fc

    # Shutdown instruments
    await fc._shutdown()

    # Shutdown async worker gracefully
    try:
        fc._worker_task.cancel()
    except asyncio.CancelledError:
        await fc._worker_task


@pytest.mark.fluidic
@pytest.mark.asyncio
@pytest.mark.slow
class TestFlowCell:
    async def test_init(self, fc: FlowCell):
        fc.initialize()
        fc.configure(DEFAULT_CONFIG)
        await fc._queue.join()

        for i in fc.iter_instruments:
            assert i.connected

        assert fc.Pump.max_volume > fc.Pump.min_volume
        assert fc._inlet in [2, 8]

    async def test_pump(self, fc: FlowCell):
        port = fc.Valve.config["port"]["valid_list"][1]
        vol = fc.Pump.min_volume * 1000
        flow = fc.Pump.min_flow_rate * 10

        fc.pump(vol, flow, reagent=port)
        await fc._queue.join()
        assert fc.Pump.ready

        port = fc.Valve.config["safe_port"]
        fc.pump(vol, flow, port, reverse=True)
        await fc._queue.join()
        assert fc.Pump.ready
