from pyseq_core.base_instruments import BaseInstrument
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re

LOGGER = logging.getLogger("PySeq")

"""
FPGA Commands

RESET --> Initialize/Reset FPGA

"""


@define(kw_only=True)
class EmulatedFPGA(EmulatedSerialCOM):
    name: str = field(default="FPGA")
    pattern: re.Pattern = field(default=re.compile(r"RESET"))

    async def command(self, command: str, read: bool = True, delay: int = 0) -> str:
        """
        Asynchronously emulate sending commands and receiving response from the FPGA.

        Args:
            command (str): The command string to be sent.
            read (bool): Whether to read a response after sending the command.
            delay (int,float): Delay before reading response.
        """

        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")

            match = re.search(self.pattern, command)

            if match:
                self.initialized = True
                response = "FPGA initialized"
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response


@define(kw_only=True)
class FPGA(BaseInstrument):
    """FPGA which controls other instruments.

    The FPGA controls the LEDs, filters, shutter, tilt motors, and the
    objective position. This class is not meant control to these instruments.
    Use the instrument specific class to control them instead. This class is
    only meant to initialize/reset the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific pump instance.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.

    """

    name: str = field(default="FPGA")
    _status: str = field(init=False)

    async def initialize(self):
        """Reset the FPGA."""
        if not self._status:
            await self.reset()

    async def configure(self):
        """No configuration needed."""
        pass

    async def shutdown(self):
        """No shutdown needed."""
        pass

    async def status(self) -> bool:
        """True if FPGA has been initialize, false otherwise.

        Returns:
            bool: True if FPGA has been initialized.
        """
        return self._status

    async def reset(self) -> bool:
        await self.command("RESET", read=2, delay=2)
        self._status = True
