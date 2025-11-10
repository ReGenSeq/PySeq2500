from pyseq_core.base_system import BaseSequencer
from pyseq2500.flowcell import FlowCell
from pyseq2500.microscope import Microscope
from pyseq2500.utils import HW_CONFIG
from attrs import define, field


@define
class Sequencer(BaseSequencer):
    name: str = field(default=HW_CONFIG["name"])
    _flowcells: dict[str, FlowCell] = field(init=False)
    _microscope: Microscope = field(factory=Microscope)

    @_flowcells.default
    def set_flowcells(self):
        return {fc: FlowCell(name=f"FlowCell{fc}") for fc in HW_CONFIG["flowcells"]}

    # async def _configure(self, exp_config: dict):
    #     self.flowcells = {"A": FlowCell("FlowCellA"),
    #                       "B": FlowCell("FlowCellB")}

    def custom_roi_stage(flowcell, **kwargs):
        pass
