from pyseq_core.base_instruments import BaseStage
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio

LOGGER = logging.getLogger("PySeq")

"""
Z -> Initialize Stage
D -> Set position
R(IP) -> Report 1 if stage is in position, 0 if not
R(PA) -> Report absolute position of stage
GAINS -> Print current gain setings
V -> Print current velocity
"""


@define(kw_only=True)
class EmulatedYStage(EmulatedSerialCOM):
    name: str = field(default="YStage")
    home_position: int = field(default=0)
    _position: int = field(default=0)
    _velocity: float = field(default=1.0)
    _gains: str = field(default="5,10,7,1.5,0")
    toggle: int = field(default=1)
    z_pattern: re.Pattern = field(default=re.compile(r"Z"))
    d_pattern: re.Pattern = field(default=re.compile(r"D(\d+)"))
    pa_pattern: re.Pattern = field(default=re.compile(r"R\(PA\)"))
    ip_pattern: re.Pattern = field(default=re.compile(r"R\(IP\)"))
    gains_pattern: re.Pattern = field(default=re.compile(r"GAINS(?:\(([\d.,]+)\))?"))
    v_pattern: re.Pattern = field(default=re.compile(r"V(\d*\.?\d*)"))

    async def command(self, command: str, read: bool = True) -> str:
        """
        Asynchronously emulate sending commands and receiving response from YStage.

        Args:
            command (str): The command string to be sent.
        """

        async with self.lock:
            cmdid = await self.write(command)

            z_match = re.search(self.z_pattern, command)
            d_match = re.search(self.d_pattern, command)
            pa_match = re.search(self.pa_pattern, command)
            ip_match = re.search(self.ip_pattern, command)
            gains_match = re.search(self.gains_pattern, command)
            v_match = re.search(self.v_pattern, command)

            if z_match:
                response = "\n"
                LOGGER.debug(f"{self.name} :: *ViX250IH-Servo Drive")
                LOGGER.debug(f"{self.name} :: *REV 2.4 Jun 29 2005 16:58:18")
                LOGGER.debug(f"{self.name} :: *Copyright 2003 Parker-Hannifin")
            elif d_match:
                response = self.d(d_match)
            elif pa_match:
                response = f"*{self._position}"
            elif ip_match:
                response = self.ip()
            elif gains_match:
                response = self.gains(gains_match)
            elif v_match:
                response = self.velocity(v_match)
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def d(self, match: re.Match) -> str:
        """Set position."""
        self._position = int(match.groups()[0])
        self.toggle = 0  # reset toggle for ip response
        return ""

    def ip(self) -> str:
        """Print 1 if stage in position, 0 if not in position."""

        # emulate waiting for stage to reach position by toggling value
        self.toggle += 1
        if self.toggle > 1:
            return "*1"
        else:
            return "*0"

    def gains(self, match: re.Match) -> str:
        """Get or set gain settings."""
        _g = match.groups()[0]
        if _g is None:
            # Get gains
            g = [float(g) for g in self._gains.split(",")]
            return f"*GF{g[0]:.2f} GI{g[1]:.2f} GP{g[2]:.2f} GV{g[3]:.2f} FT{g[4]:.0f}"
        else:
            # set gains
            self._gains = _g
            return ""

    def velocity(self, match: re.Match) -> str:
        """Get or set velocity."""
        v = match.groups()[0]
        if len(v) == 0:
            # Get velocity
            return f"*{self._velocity:3f}"
        else:
            # Set velocity
            self._velocity = float(v)
            return ""


@define(kw_only=True)
class YStage(BaseStage):
    name: str = field(default="YStage")
    home_position: int = field(default=0)
    _mode: str = field(default="")

    async def initialize(self):
        await self.command("Z", read=False)  # Initialize/Reset Stage
        await asyncio.sleep(2)  # Wait 2 s for stage to reset
        await self.com.read()  # read driver name
        await self.com.read()  # read driver version
        await self.com.read()  # read copyright
        await self.com.read()  # read new line
        await self.command("W(EX,0)")  # Turn off echo mode
        await self.set_mode("moving")  # Set gains and velocity to moving mode
        await self.com.write("MA")  # Set to absolute positioning mode
        await self.com.write("ON")  # Turn on motor
        await self.com.write("GH")  # Home Stage
        while not await self.status():
            await asyncio.sleep(1)

    async def configure(self):
        """Configure position limits on XStage."""
        # Already implemented in pyseq_core
        pass

    async def shutdown(self):
        """Return stage to home position."""
        await self.move(self.home_position)  # Move to home position

    async def status(self) -> bool:
        """Query stage for status and position.

        Returns:
            bool: True if the stage is in position, false otherwise.
        """
        response = await self.command("R(IP)")  # Query if stage is in position
        await self.get_position()

        return int(response[1:]) == 1

    async def move(self, position: int):
        """Move the stage to a specified step position.
        Args:
            position (Union[int, float]): The target position to move the stage to.
        """
        if position != self.position:
            await self.command(f"D{position}", read=False)  # Set position
            await self.com.write("G")  # Go to position
            while not await self.status():
                await asyncio.sleep(1)

    async def get_position(self):
        """Retrieve the current actual position of the stage.

        Returns:
            Union[int, float]: The current position of the stage.
        """

        p = await self.command("R(PA)")
        self.position = int(p[1:])

        return self.position

    async def set_mode(self, mode: str):
        """Set the mode for the YStage.

        Returns:
            bool: True if mode was set succesfully.
        """

        if mode not in self.config["mode"].keys():
            raise ValueError(f"Invalid mode {mode} for YStage")
            return False
        else:
            gains = self.config["mode"][mode]["gains"]
            velocity = self.config["mode"][mode]["velocity"]

        while self._mode != mode:
            if await self.set_gains(gains) and await self.set_velocity(velocity):
                self._mode = mode

        return True

    async def set_gains(self, gains: str):
        """Set the gain settings for the YStage.

        Returns:
            bool: True if gains were set succesfully.
        """

        await self.command(f"GAINS({gains})", read=False)
        response = await self.command("GAINS")
        response = response.strip()[1:].split(" ")
        _gains = [float(g) for g in gains.split(",")]
        all_true = all([float(g[2:]) == _gains[i] for i, g in enumerate(response)])

        return all_true

    async def set_velocity(self, velocity: float):
        """Set the velocity for the YStage.

        Returns:
            bool: True if velocity was set successfully.
        """

        await self.command(f"V{velocity}", read=False)
        response = await self.command("V")
        response = float(response.strip()[1:])

        return response == velocity
