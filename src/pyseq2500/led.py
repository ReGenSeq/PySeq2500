from pyseq_core.base_instruments import BaseInstrument
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re

LOGGER = logging.getLogger("PySeq")

"""
Status Light Commands

LEDMODE fc x --> set LED on flowcell fc to color x
LEDSWPRATE x --> set LED sweep rate to x
LEDPULSRATE x --> set LED pulse rate to x

"""

COLOR_DICT = {
    "off": 0,
    "yellow": 1,
    "green": 3,
    "pulse green": 4,
    "blue": 5,
    "pulse blue": 6,
    "sweep blue": 7,
    "imaging": 7,  # sweep blue
    "pumping": 6,  # pulse blue
    "waiting": 5,  # blue
    "standby": 3,  # green
    "user": 4,  # pulse green
    "error": 1,  # yellow
}


@define(kw_only=True)
class EmulatedLED(EmulatedSerialCOM):
    name: str = field(default="FPGA")
    color_id: int = field(init=False)
    pattern: re.Pattern = field(default=re.compile(r"LEDMODE(\d) (\d)"))

    async def command(self, command: str, read: bool = True, delay: int = 0) -> str:
        """
        Asynchronously emulate sending commands and receiving response from LED via FPGA.

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
                fc, self.color_id = match.groups()
                response = f"LEDMODE{fc}"
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response


@define(kw_only=True)
class LED(BaseInstrument):
    """LED indicator light for a flowcell controlled through the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific LED instance LEDA or LEDB.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.
    """

    id: int = field(init=False)
    _status: str = field(init=False)

    async def initialize(self):
        """Reset the FPGA."""
        await self.set_led("standby")

    async def configure(self):
        """Configure ID based on flowcell and update color dict."""
        self.id = 1 if self.name[-1] == "A" else 2

    async def shutdown(self):
        """Turn laser off."""
        await self.set_led("off")

    async def status(self) -> bool:
        """True if commands are received by FPGA"""
        return self._status

    async def set_led(self, color: str, sweep: int = None, pulse: int = None):
        """Set LED color/mode and optionally sweep/pulse rate.
        **Available Colors/Modes:**
         - off
         - yellow
         - green
         - pulse green
         - blue
         - pulse blue
         - sweep blue
         - imaging (sweep blue)
         - pumping (pulse blue)
         - waiting (blue)
         - standby (green)
         - user (pulse green)
         - error (yellow)

         Args:
         color (str): Color to set the LED to.
         sweep (1-255): Optional sweep rate
         pulse (1-255): Optional pulse rate
        """

        mode = COLOR_DICT.get(color, None)

        if mode is not None:
            response = await self.command(f"LEDMODE{self.id} {mode}")
            self._status = response.strip() == f"LEDMODE{self.id}"
        else:
            LOGGER.warning(f"{self.name} :: {color} is not available")

        if sweep is not None and 1 <= sweep <= 255:
            await self.command("LEDSWPRATE {sweep}")

        if pulse is not None and 1 <= pulse <= 255:
            await self.command("LEDPULSRATE {pulse}")
