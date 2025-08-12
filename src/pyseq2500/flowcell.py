from pyseq_core.base_system import BaseFlowCell
from pyseq2500.fluidics import Pump, Valve
from pyseq2500.com import COM_DICT
from typing import Literal
import logging
from attrs import define, field

LOGGER = logging.getLogger("PySeq")


@define
class FlowCell(BaseFlowCell):
    _inlet: int = field(init=False)
    instruments: dict = field(init=False)

    @property
    def position(self):
        return self.name[-1:]

    @instruments.default
    def set_instruments(self):
        fc = self.position
        self.instruments = {
            "Pump": Pump(name="Pump{fc}", com=COM_DICT[f"Pump{fc}"]),
            "Valve": Valve(name="Valve24{fc}", com=COM_DICT[f"Valve24{fc}"]),
            "InletValve": Valve(name="Valve10{fc}", com=COM_DICT[f"Valve10{fc}"]),
        }

    @property
    def InletValve(self) -> Valve:
        return self.instruments["InletValve"]

    @property
    def inlet(self):
        return self._inlet

    @inlet.setter
    def inlet(self, inlet: Literal[2, 8]):
        if inlet in [2, 8] and self._inlet != inlet:
            description = "Selecting {inlet} port inlet"
            self.add_task(description, self._select_inlet, inlet)
        elif inlet not in [2, 8]:
            raise ValueError("Invalid inlet {inlet}, select or 2 or 8 port inlet")

    async def _select_inlet(self, inlet: Literal[2, 8]):
        port = self._config["inlet_port"][inlet]
        await self.InletValve.select(port)
        self._inlet = inlet

    async def _configure(self, exp_config: dict):
        """Configure the system."""

        # Update pump limits
        await self.Pump.configure(exp_config=exp_config)
        # Set 2 or 8 port inlet
        inlet = exp_config["flowcell"]["inlet"]
        await self._select_inlet(inlet)
        self._inlet = inlet
