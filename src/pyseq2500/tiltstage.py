import logging
from pyseq_core.base_instruments import BaseStage
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio
from typing import Union
from functools import cached_property

LOGGER = logging.getLogger("PySeq")

"""
3 tilt motors (1,2,3) to control large z movements and tilt of the stage. 
    
TiMOVETOx           -> Move tilt i to position x
TiRDi          -> Read position of tilt i
TiHM             -> Home tilt i, home
TiCR           -> Clear the motor count registers (use after TiHM)
"""


@define(kw_only=True)
class EmulatedTiltMotor(EmulatedSerialCOM):
    _position: str = field(default="DISABLED")
    move_pattern: re.Pattern = field(default=re.compile(r"T(\d)MOVETO (\d+)"))
    rd_pattern: re.Pattern = field(default=re.compile(r"T(\d)RD"))
    hm_pattern: re.Pattern = field(default=re.compile(r"T(\d)HM"))
    cr_pattern: re.Pattern = field(default=re.compile(r"T(\d)CR"))

    @cached_property
    def id(self):
        return self.name[-1]

    async def command(self, command: str, read: bool = True) -> str:
        """
        Asynchronously emulate sending commands and receiving response from TiltMotor.

        Args:
            command (str): The command string to be sent.
        """

        async with self.lock:
            cmdid = await self.write(command)

            move_match = re.search(self.move_pattern, command)
            rd_match = re.search(self.rd_pattern, command)
            hm_match = re.search(self.hm_pattern, command)
            cr_match = re.search(self.cr_pattern, command)

            if move_match:
                self.move(move_match)
                response = command
            elif rd_match:
                response = f"T{self.id}RD {self._position}"
            elif hm_match:
                self._position = 1
                response = command
            elif cr_match:
                self._position = 0
                response = command
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def move(self, match: re.Match):
        """Move to position"""
        position = match.groups()[1]
        self._position = int(position)


@define(kw_only=True)
class TiltMotor(BaseStage):
    """Concrete implementation for the Tilt Motor.

    This class provides a specific implementation for the abstract methods
    defined in `BaseStage`, allowing for control of the tilt motor.
    It handles communication with the motor to move and read the position of a
    tilt motor.

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

    @cached_property
    def id(self):
        return self.name[-1]

    async def initialize(self):
        """Initialize and home the Tilt Motor."""
        self.position = 1
        while self.position != 0:
            await self.home()
            await self.status()
            await self.clear()
            await asyncio.sleep(0.1)
            await self.get_position()

    async def home(self):
        """Home motor."""
        # Need to read 3 lines
        await self.command(f"T{self.id}HM", read=2, delay=2)

    async def clear(self):
        "Clear motor register."
        await self.command(f"T{self.id}CR", delay=1)

    async def configure(self):
        """Configure position limits on Tilt Motor."""
        # Already implemented in pyseq_core
        pass

    async def shutdown(self):
        """Return motor to home position."""
        await self.home()

    async def status(self):
        """Waits for motor to finish moving before returning True"""

        new_position = await self.get_position()
        old_position = new_position + 1
        while old_position != new_position:
            old_position = new_position
            await asyncio.sleep(1)
            new_position = await self.get_position()
        return True

    async def move(self, position: int):
        """Move the motor to the specified position.

        Args:
            position (int): The target position to move the motor to.
        """

        while abs(self.position - position) >= self.tolerance:
            await self.command(f"T{self.id}MOVETO {position}", delay=1)
            await self.status()

    async def get_position(self):
        """Retrieve the current actual position of the motor.

        Returns:
            int: The current position of the motor.
        """
        position = await self.command(f"T{self.id}RD")
        while len(position.strip()) == 0:
            position = await self.com.read()
            await asyncio.sleep(0.1)
        self._position = int(position[5:])

        return self._position

    @cached_property
    def tolerance(self) -> int:
        return self.config["tolerance"]


@define(kw_only=True)
class TiltStage(BaseStage):
    """Control all 3 Tilt Motors to tilt and move the stage.


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

    name: str = field(default="TiltStage")
    tilts: dict[int, TiltMotor] = field(default={1: None, 2: None, 3: None})

    async def initialize(self):
        """Initialize and home the Tilt Motor."""

        # Initialize FPGA
        if self.com._cmdid == 0:
            await self.command("RESET\n", read=2, delay=2)

        # Assign motors
        for i in range(1, 4):
            if self.tilts[i] is None:
                self.tilts[i] = TiltMotor(name=f"TiltMotor{i}", com=self.com)

        # Initialize tilts
        _ = []
        for tilt in self.tilts.values():
            _.append(tilt.initialize())
        await asyncio.gather(*_)

        # Update positions
        self.update_positions()

    async def configure(self):
        """Configure position limits on Tilt Motor."""
        # Already implemented in pyseq_core
        pass

    async def shutdown(self):
        """Return motor to home position."""
        _ = []
        for tilt in self.tilts.values():
            _.append(tilt.shutdown())
        await asyncio.gather(*_)

    async def status(self):
        """Wait for motors to finish moving before returning True"""

        # Check to see motors are stopped
        _ = []
        for tilt in self.tilts.values():
            _.append(tilt.status())
        _status = await asyncio.gather(*_)

        self.update_positions()

        return all(_status)

    async def move(self, position: Union[int, list[int]]):
        """Move the motors to the specified position.

        Args:
            position (int): The target position to move the motor to.
        """

        if isinstance(position, int):
            position = [position] * 3

        # Move motors
        _ = []
        for tilt, pos in zip(self.tilts.values(), position):
            _.append(tilt.move(pos))
        await asyncio.gather(*_)

        self.update_positions()

    async def get_position(self):
        """Retrieve the current actual position of the motor.

        Returns:
            int: The current position of the motor.
        """

        _ = []
        for tilt in self.tilts.values():
            _.append(tilt.get_position())
        self._position = asyncio.gather(*_)

    def update_positions(self):
        """Update cached positions"""
        _ = []
        for tilt in self.tilts.values():
            _.append(tilt.position)
        self._position = _
