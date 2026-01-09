from pyseq_core.base_instruments import BaseFilter, BaseShutter
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re

LOGGER = logging.getLogger("PySeq")

"""
FPGA Optical Commands

EX{l}MV X    --> Move excitation filter wheel l to position X
EX{l}HM      --> Home excitation filter wheel l 
EM2X         --> Move emission filter wheel to X (I for in, or O for out)
SWLSRSHUT X  --> Close (X=0) or open (X=1) laser shutter
"""


@define(kw_only=True)
class EmulatedOptics(EmulatedSerialCOM):
    name: str = field(default="FPGA")
    exmv_pattern: re.Pattern = field(default=re.compile(r"EX([12])MV (-?\d+)"))
    exhm_pattern: re.Pattern = field(default=re.compile(r"EX([12])HM"))
    em_pattern: re.Pattern = field(default=re.compile(r"EM2([IO])"))
    shut_pattern: re.Pattern = field(default=re.compile(r"SWLSRSHUT ([01])"))

    async def command(self, command: str, read: bool = True, delay: int = 0) -> str:
        """
        Asynchronously emulate sending commands and receiving response from filters via FPGA.

        Args:
            command (str): The command string to be sent.
            read (bool): Whether to read a response after sending the command.
            delay (int,float): Delay before reading response.
        """

        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")

            exmv_match = re.search(self.exmv_pattern, command)
            exhm_match = re.search(self.exhm_pattern, command)
            em_match = re.search(self.em_pattern, command)
            shut_match = re.search(self.shut_pattern, command)

            if exmv_match:
                response = self.exmv(exmv_match)
            elif exhm_match:
                response = command
            elif em_match:
                response = command
            elif shut_match:
                response = "SWLSRSHUT"
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def exmv(self, match: re.Match) -> str:
        id, position = match.groups()
        return f"EX{id}MV"


@define(kw_only=True)
class FilterWheel(BaseFilter):
    """Optical filter wheel controlled through the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific filter, Green or Red.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BaseFilter Attributes:
        _filters (dict): A dictionary mapping filter names to their positions on
            the filter wheel
        _filter (Union[float, str]): The cached currently selected filter on the
            wheel.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.

    """

    id: int = field(init=False)

    async def initialize(self):
        """Reset the FPGA."""
        await self.home()

    async def configure(self):
        """Configure ID based on flowcell and update color dict."""
        self.id = 1 if self.name[1] == "G" else 2

    async def shutdown(self):
        """Turn laser off."""
        await self.home()

    async def status(self) -> bool:
        """Just returns True"""
        return True

    async def home(self):
        await self.command(f"EX{self.id}HM")
        self._filter = "home"

    async def set_filter(self, filter):
        """Set LED color/mode and optionally sweep/pulse rate.

        Args:
            filter (str): Filter to turn the wheel to
        """

        position = self._filters.get(filter, None)

        if filter != self.filter and position is not None:
            await self.home()
            if filter != "home":
                await self.command(f"EX{self.id}MV {position}")
            self._filter = filter
        else:
            LOGGER.warning(f"{self.name} :: {filter} is not available")

    async def get_filter(self, filter):
        """Not implemented for filter wheel."""
        return self.filter


@define(kw_only=True)
class EmissionFilter(BaseFilter):
    """Emission filter controlled through the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific filter, Green or Red.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BaseFilter Attributes:
        _filters (dict): A dictionary mapping filter names to their positions on
            the filter wheel
        _filter (Union[float, str]): The cached currently selected filter on the
            wheel.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.

    """

    name: str = field(default="EmissionFilter")
    _filter: str = field(default="")

    async def initialize(self):
        """Move filter into emission light path."""
        await self.set_filter("in")

    async def configure(self):
        """No configuration needed."""
        pass

    async def shutdown(self):
        """Turn laser off."""
        await self.set_filter("in")

    async def status(self) -> bool:
        """Just returns True"""
        return True

    async def set_filter(self, filter):
        """Move emission filter in or out of light path.

        Args:
            filter (str): in or out
        """

        filter = filter.lower()
        position = None
        if "i" in filter:
            position = "I"
        elif "o" in filter:
            position = "O"
        else:
            LOGGER.warning(f"{self.name} :: {filter} is not available")

        if position is not None and filter != self.filter:
            response = await self.command(f"EM2{position}")
            if response.strip() == f"EM2{position}":
                self._filter = filter

    async def get_filter(self, filter):
        """Return cached filter."""
        return self.filter


@define(kw_only=True)
class Shutter(BaseShutter):
    """Shutter controlled through the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific filter, Green or Red.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BaseShutter Attributes:
        _open (bool): An cached attribute indicating whether the shutter is
            currently open (`True`) or closed (`False`). This should be accessed
            via the `is_open` property.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.
    """

    name: str = field(default="Shutter")

    async def initialize(self):
        """Close Shutter"""
        self._open = True
        await self.close()

    async def configure(self):
        """No configuration needed."""
        pass

    async def shutdown(self):
        """Close shutter"""
        await self.close()

    async def status(self) -> bool:
        """True if open, False if closed"""
        return self.is_open

    async def open(self):
        """Send command to open shutter."""
        if not self.is_open:
            response = await self.command("SWLSRSHUT 1")
            self._open = response.strip() == "SWLSRSHUT"

    async def close(self):
        """Send command to open shutter."""
        if self.is_open:
            response = await self.command("SWLSRSHUT 0")
            self._open = not (response.strip() == "SWLSRSHUT")
