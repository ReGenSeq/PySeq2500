from pyseq_core.base_instruments import BaseInstrument
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio
from typing import Optional

LOGGER = logging.getLogger("PySeq")

"""
FPGA Commands

RESET --> Initialize/Reset FPGA
TDIYERD --> Read y stage position
TDIYEWR X --> Write y stage position as X
TDIYPOS X --> Arm camera trigger to start at YStage position X
TDIYARM3 X Y 1 --> Arm camera trigger to take X frames at YStage position Y

"""


@define(kw_only=True)
class EmulatedFPGA(EmulatedSerialCOM):
    name: str = field(default="FPGA")
    position: int = field(default=0)
    initialized: bool = field(default=False)
    reset_pattern: re.Pattern = field(default=re.compile(r"RESET"))
    read_pattern: re.Pattern = field(default=re.compile(r"TDIYERD"))
    write_pattern: re.Pattern = field(default=re.compile(r"TDIYEWR (\d+)"))
    ypos_pattern: re.Pattern = field(default=re.compile(r"TDIYPOS (\d+)"))
    yarm_pattern: re.Pattern = field(default=re.compile(r"TDIYARM3 (\d+) (\d+) 1"))

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

            reset_match = re.search(self.reset_pattern, command)
            read_match = re.search(self.read_pattern, command)
            write_match = re.search(self.write_pattern, command)
            ypos_match = re.search(self.ypos_pattern, command)
            yarm_match = re.search(self.yarm_pattern, command)

            if reset_match:
                self.initialized = True
                response = "RESET"
            elif read_match:
                response = f"TDIYERD {self.position}"
            elif write_match:
                response = self.write(write_match)
            elif ypos_match:
                response = "TDIYPOS"
            elif yarm_match:
                response = "TDIYARM3"
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def write(self, match: re.Match):
        self.position = int(match.groups()[0])
        return f"TDIYEWR {self.position}"


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
    _status: bool = field(default=False)
    _position: Optional[int] = field(default=None)

    @property
    def y_offset(self):
        return 7000000

    async def initialize(self):
        """Reset the FPGA."""
        if not self._status:
            await self.reset()

    async def configure(self, exp_config: dict = {}):
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
        response = await self.command("RESET", read=2, delay=1)
        self._status = response.strip() == "RESET"
        if self._position is not None:
            await self.write_position(self._position)

    async def read_position(self) -> int:
        tdi_pos = None
        while not isinstance(tdi_pos, int):
            try:
                response = await self.command("TDIYERD")
                response = response.split(" ")[1].strip()
                tdi_pos = int(response) - self.y_offset
            except ValueError:
                tdi_pos = None
                await asyncio.sleep(1)
        return tdi_pos

    async def write_position(self, position: int):
        while abs(await self.read_position() - position) > 5:
            await self.command(f"TDIYEWR {position + self.y_offset}")
            self._position = position

    async def TDIARM(self, y_init: int, n_triggers: int):
        confirmed = False
        while not confirmed:
            y_pos = y_init + self.y_offset - 80000
            response = await self.command(f"TDIYPOS {y_pos}", delay=1)
            if response.strip() == "TDIYPOS":
                confirmed = True
            else:
                await self.reset()

            if confirmed:
                y_pos = y_init + self.y_offset - 10000
                response = await self.command(f"TDIYARM3 {n_triggers} {y_pos} 1")
                if response.strip() == "TDIYARM3":
                    confirmed = True
                else:
                    confirmed = False
                    await self.reset()
