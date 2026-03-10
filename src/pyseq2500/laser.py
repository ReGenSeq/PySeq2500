from pyseq_core.base_instruments import BaseLaser
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re

LOGGER = logging.getLogger("PySeq")

"""
STAT?         -> Returns the status of the laser (ENABLED/DISABLED)
ON            -> Turns the laser on
OFF           -> Turns the laser off
POWER=X       -> Sets the power of the laser to X mW
POWER?        -> Returns the current power of the laser in mW
"""


@define(kw_only=True)
class EmulatedLaser(EmulatedSerialCOM):
    _status: str = field(default="DISABLED")
    _power: int = field(default=0)
    stat_pattern: re.Pattern = field(default=re.compile(r"STAT\?"))
    on_pattern: re.Pattern = field(default=re.compile(r"\bON\b"))
    off_pattern: re.Pattern = field(default=re.compile(r"OFF"))
    power_pattern: re.Pattern = field(default=re.compile(r"POWER=?(\d*)?"))

    async def command(self, command: str, read: bool = True) -> str:
        """
        Asynchronously emulate sending commands and receiving response from laser.

        Args:
            command (str): The command string to be sent.
            read (bool): Whether to read a response after sending the command.
        """

        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")

            stat_match = re.search(self.stat_pattern, command)
            on_match = re.search(self.on_pattern, command)
            off_match = re.search(self.off_pattern, command)
            power_match = re.search(self.power_pattern, command)

            if stat_match:
                response = self.stat()
            elif on_match:
                response = self.on()
            elif off_match:
                response = self.off()
            elif power_match:
                response = self.power(power_match)
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def stat(self) -> str:
        return self._status

    def on(self) -> str:
        self._status = "ENABLED"
        return ""

    def off(self) -> str:
        self._status = "DISABLED"
        return ""

    def power(self, match: re.Match) -> str:
        g = match.groups()[0]
        if len(g) > 0:
            self._power = int(g)
            return ""
        else:
            return f"{self._power:04d}mW"


@define(kw_only=True)
class Laser(BaseLaser):
    """Concrete implementation of a laser instrument.

    This class provides a specific implementation for the abstract methods
    defined in `BaseLaser`, allowing for controlling the power of a laser..

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific pump instance.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BasePump Attributes:
        color (str): The color of the laser beam (e.g., "red", "green", "blue").
        _power (Union[int, float]): The current cached power setting of the
            laser. This attribute is not initialized directly but is set by the
            `power` setter.

    Laser Attributes:
        _status (str): True if the laser is on, False if off.

    Inherited BaseInstrument Methods:
        command(str) -> str: Send a command to the pump.
    """

    _status: bool = field(init=False)

    async def initialize(self):
        """Turn laser on and set to low power."""
        await self.command("VERSION?", read=2)  # Get firmware version
        await self.status()  # Get initial status
        await self.get_power()  # Get initial power

    async def configure(self, exp_config: dict = {}):
        # nothing to configure for laser
        pass

    async def shutdown(self):
        """Turn laser off."""
        await self.command("OFF")  # turn off the laser

    async def status(self) -> bool:
        """Query laser for status and power.

        Returns:
            bool: True if the laser is enabled.
        """

        status = await self.command("STAT?")
        self._status = "ENABLED" == status.strip()
        if self._status:
            await self.get_power()
            return self._power > 0
        else:
            return False

    async def set_power(self, power):
        """Sets the power of the laser.

        Args:
            power (Union[int, float]): The desired power level to set.
        """

        if not self._status:
            await self.command("ON")  # turn on the laser
            await self.status()  # update status

        if self.power != power:
            await self.command(f"POWER={power}")
        await self.get_power()

    async def get_power(self):
        """Gets the current power of the laser.

        Returns:
            int: The current power level of the laser.
        """
        response = await self.command("POWER?")
        self._power = int(response.split("mW")[0])
        if self._status and self.power < 0:
            LOGGER.warning(f"{self.color} Laser is at 0 mW power.")

        return self._power
