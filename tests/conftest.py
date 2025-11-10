import pytest_asyncio
import pytest
from pyseq2500.flowcell import FlowCell
from pyseq2500.fluidics import Pump, Valve, EmulatedPump, EmulatedValve
from pyseq2500.utils import HW_CONFIG

# @pytest.mark.hardware
# @pytest_asyncio.fixture
# async def Sequencer():
#     """Test Sequencer with default settings."""

#     # Sequencer Setup
#     from pyseq2500.sequencer import Sequencer
#     seq = Sequencer()
#     seq.start()
#     seq.initialize()
#     await seq._queue.join()

#     # Sequencer call
#     yield seq

#     # Sequencer Teardown
#     # Shutdown systems and cancel task workers
#     seq.shutdown()


@pytest_asyncio.fixture(
    params=[
        pytest.param("MockSequencer", marks=pytest.mark.mock),
        pytest.param("TestSequencer", marks=pytest.mark.hardware),
    ],
    scope="module",
)
async def sequencer(request):
    from pyseq2500.sequencer import Sequencer

    name = request.param
    seq = Sequencer(name=name)

    if name == "MockSequencer":
        flowcells = {}
        for fc in HW_CONFIG["flowcells"]:
            pcom = EmulatedPump(name=f"Pump{fc}", address=f"Pump{fc}COM")
            v24com = EmulatedValve(name=f"Valve24{fc}", address=f"Valve24{fc}COM")
            v10com = EmulatedValve(name=f"Valve10{fc}", address=f"Valve10{fc}COM")

            instruments = {
                "Pump": Pump(name=f"Pump{fc}", com=pcom),
                "Valve": Valve(name=f"Valve24{fc}", com=v24com),
                "InletValve": Valve(name=f"Valve10{fc}", com=v10com),
            }
            flowcell = FlowCell(name=f"FlowCell{fc}")
            flowcell.instruments = instruments
            flowcells[fc] = flowcell

        seq._flowcells = flowcells

    # seq.start()
    # seq.initialize()
    # await seq._queue.join()

    # Sequencer call
    yield seq

    # Sequencer Teardown
    # Shutdown systems and cancel task workers
    # seq.shutdown()
