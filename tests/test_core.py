from pyseq_core.utils import parse
from pyseq2500.fluidics import PumpStatus, PUMP_STATUS


def test_parse():
    assert parse(PUMP_STATUS, "0`0", PumpStatus)
