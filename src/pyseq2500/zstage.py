import logging
from pyseq_core.base_instruments import BaseStage
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio
from functools import cached_property

LOGGER = logging.getLogger("PySeq")

"""
Control Objective height and triggers
    
ZMV X     -> Move objective to position x
ZDACR     -> Read position of objective
ZSTEP X   -> Set velocity X (steps/s)
ZTRG X    -> Trigger camera at step X
ZYT 0 3   -> Set camera to be triggered by objective (use after ZTRG X)
"""


@define(kw_only=True)
class EmulatedZStage(EmulatedSerialCOM):
    name: str = field(default="FPGA")
    _position: str = field(default=30000)
    move_pattern: re.Pattern = field(default=re.compile(r"ZMV (\d+)"))
    read_pattern: re.Pattern = field(default=re.compile(r"ZDACR"))
    step_pattern: re.Pattern = field(default=re.compile(r"ZSTEP (\d+)"))

    async def command(self, command: str, read: bool = True, delay: int = 0) -> str:
        """
        Asynchronously emulate sending commands and receiving response from Z Stage.

        Args:
            command (str): The command string to be sent.
        """

        async with self.lock:
            cmdid = await self.write(command)

            move_match = re.search(self.move_pattern, command)
            read_match = re.search(self.read_pattern, command)
            step_match = re.search(self.step_pattern, command)

            if move_match:
                response = self.move(move_match)
            elif read_match:
                response = self.read()
            elif step_match:
                response = "ZSTEP"
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def move(self, match: re.Match):
        """Move to position"""
        position = match.groups()[0]
        self._position = int(position)
        return "ZMV"

    def read(self):
        """Read position"""
        return f"ZDACR {self._position}"


@define(kw_only=True)
class ZStage(BaseStage):
    """ZStage to control the objective positioning.

    This class provides a specific implementation for the abstract methods
    defined in `BaseStage`, allowing for control of the ZStage through the FPGA.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific pump instance.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BasePump Attributes:
        _position (Union[int, float]): The cached current position of the motor.
            This attribute is not initialized directly but is set by the `position` setter.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the motor.
    """

    name: str = field(default="ZStage")
    _status: bool = field(init=False)
    _velocity: int = field(init=False)

    async def initialize(self):
        """Update position and velocity of objective"""
        await self.get_position()
        await self.set_velocity(self.max_velocity)

    async def configure(self):
        """Configure focus on Tilt Motor."""
        pass

    async def shutdown(self):
        """Nothing to shutdown"""
        pass

    async def status(self):
        """Return cached status of objective."""
        return self._status

    async def move(self, position: int):
        """Move the motor to the specified position.

        Args:
            position (int): The target position to move the motor to.
        """

        while self.position != position:
            response = await self.command(f"ZMV {position}")
            if response.strip() == "ZMV":
                await asyncio.sleep(1)
                await self.get_position()
            else:
                self._status = False

    async def get_position(self):
        """Retrieve the current actual position of the motor.

        Returns:
            int: The current position of the motor.
        """
        position = await self.command("ZDACR")
        try:
            self._position = int(position.split(" ")[1])
            self._status = True
        except TypeError:
            LOGGER.warning("{self.name} :: Could not parse objective position")
            self._status = False

        return self._position

    async def set_velocity(self, velocity: float):
        """Set velocity in mm/s"""

        # Convert from mm/s to step/s
        vel = velocity * 1288471

        response = await self.command(f"ZSTEP {vel}")
        if response.strip() == "ZSTEP":
            self._velocity = velocity
            self._status = True
        else:
            self._status = False

    @property
    def velocity(self):
        """Cached velocity of the objective."""
        return self._velocity

    @cached_property
    def min_velocity(self):
        """Minimum velocity of the objective."""
        return self.config["velocity"]["min_val"]

    @cached_property
    def max_velocity(self):
        """Maximum velocity of the objective."""
        return self.config["velocity"]["max_val"]
