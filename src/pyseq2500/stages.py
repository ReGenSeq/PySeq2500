from pyseq_core.base_instruments import BaseStage
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio

LOGGER = logging.getLogger("PySeq")

"""
PR I1 -> Print state of input 1, 1 if high and stage in position, 0 if low and stage is out of position
MA -> move to absolute position
PR MV -> Print 1 if stage is moving, 0 if not moving
PR P -> Print position of stage

"""


@define(kw_only=True)
class EmulatedXStage(EmulatedSerialCOM):
    name: str = field(default="XStage")
    home_position: int = field(default=30000)
    position: int = field(default=30000)
    toggle: int = field(default=1)
    i1_pattern: re.Pattern = field(default=re.compile(r"PR I1"))
    ma_pattern: re.Pattern = field(default=re.compile(r"MA (\d+)"))
    mv_pattern: re.Pattern = field(default=re.compile(r"PR MV"))
    p_pattern: re.Pattern = field(default=re.compile(r"PR P"))
    other_pattern: re.Pattern = field(default=re.compile(r"="))
    init_pattern: re.Pattern = field(default=re.compile(r"\x03"))

    async def command(self, command: str) -> str:
        """
        Asynchronously emulate sending commands and receiving response from XStage.

        Args:
            command (str): The command string to be sent.
        """

        async with self.lock:
            cmdid = await self.write(command)

            i1_match = re.search(self.i1_pattern, command)
            ma_match = re.search(self.ma_pattern, command)
            mv_match = re.search(self.mv_pattern, command)
            p_match = re.search(self.p_pattern, command)
            init_match = re.search(self.init_pattern, command)
            other_match = re.search(self.other_pattern, command)

            if i1_match:
                response = self.i1()
            elif ma_match:
                response = self.ma(*ma_match.groups())
            elif mv_match:
                response = self.mv()
            elif p_match:
                response = self.p()
            elif other_match or init_match:
                response = ""
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""
            response = f"{self.prefix}{response}{self.suffix}"
            LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
            return response

    def i1(self) -> int:
        """Print 1 if stage in I1 position, 0 if not in I1 position."""
        if self.position == self.home_position - 1000:
            return 1
        else:
            return 0

    def ma(self, position: int) -> None:
        """Move to absolute position."""
        self.position = int(position)
        return ""

    def mv(self) -> int:
        """Print 1 if stage is moving, 0 if not moving."""
        self.toggle = -self.toggle
        if self.toggle > 0:
            return 1
        else:
            return 0

    def p(self) -> int:
        """Print position of stage."""
        return self.position


@define(kw_only=True)
class XStage(BaseStage):
    name: str = field(default="XStage")
    home_position: int = field(default=30000)

    async def initialize(self):
        """Initialize and home the XStage."""

        await self.command("\x03")  # Initialize Stage

        # Configure XStage
        await self.command(
            "EM=2"
        )  # Change echo mode to respond only to print and list commands
        await self.com.write("EE=1")  # Enable Encoder
        await self.com.write("VI=40")  # Set Initial Velocity
        await self.com.write("VM=1000")  # Set Max Velocity
        await self.com.write("A=4000")  # Set Acceleration
        await self.com.write("D=4000")  # Set Deceleration
        await self.com.write("S1=1,0,0")  # Set Home
        await self.com.write("S2=3,1,0")  # Set Neg. Limit
        await self.com.write("S3=2,1,0")  # Set Pos. Limit
        await self.com.write("SM=0")  # Set Stall Mode = stop motor
        await self.com.write("LM=1")  # Limit mode = stop if sensed
        await self.com.write("DB=8")  # Encoder Deadband
        await self.com.write("D1=5")  # Debounce home
        await self.com.write("HC=20")  # Set hold current
        await self.com.write("RC=100")  # Set run current

        """
        Home stage program
        PG 1\r                                                                  #Enter Program 1
        HM 1\r                                                                  #Homing routine 1         
        H\r                                                                     #Hold stage until home found
        P = {self.home_position}\r                                              #Set Home position       
        E\r                                                                     #End Program
        PG\r                                                                    #Exit Program
        """
        home_program = f"PG 1\rHM 1\rH\rP = {self.home_position}\rE\rPG\r"
        await self.com.write(home_program)

        # Home Stage
        homed = False
        while not homed:
            await self.com.write("EX 1")
            while not await self.status():
                # Wait until stage stops moving
                pass
            # Check if home input changes
            await self.move(self.home_position - 1000)
            homed = await self.read_input()
            await self.move(self.home_position)

        return True

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
            bool: True if the stage is not moving, false otherwise.
        """
        response = await self.command("PR MV")
        ready = not bool(int(response))

        await self.get_position()

        return ready

    async def move(self, position: int):
        """Move the stage to a specified step position.
        Args:
            position (Union[int, float]): The target position to move the stage to.
        """
        if position != self.position:
            while position != await self.get_position():
                await self.com.write(f"MA {position}")
                while await self.status():
                    await asyncio.sleep(1)

    async def get_position(self):
        """Retrieve the current actual position of the stage.

        Returns:
            Union[int, float]: The current position of the stage.
        """

        self.position = int(await self.command("PR P"))

        return self.position

    async def read_input(self, input: int = 1):
        """Read the value of the specfied input.

        Returns:
            bool: True if 1, False if 0
        """

        response = await self.command(f"PR I{input}")
        return bool(int(response.strip()))


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

    async def command(self, command: str) -> str:
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
                response = "*ViX250IH-Servo Drive"
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

            response = f"{self.prefix}{response}{self.suffix}"
            LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
            return response

    def d(self, match: re.Match) -> str:
        """Set position."""
        self._position = int(match.groups()[0])
        return ""

    def ip(self) -> str:
        """Print 1 if stage in position, 0 if not in position."""

        # emulate waiting for stage to reach position by toggling value
        self.toggle = -self.toggle
        if self.toggle > 0:
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

    def velocity(self, match: re.Match) -> str:
        """Get or set velocity."""
        v = match.groups()[0]
        if len(v) == 0:
            # Get velocity
            return f"*{self._velocity:3f}"
        else:
            # Set velocity
            self._velocity = float(v)


@define(kw_only=True)
class YStage(BaseStage):
    name: str = field(default="YStage")
    home_position: int = field(default=0)
    _mode: str = field(default="")

    async def initialize(self):
        await self.com.write("Z")  # Initialize/Reset Stage
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
            await self.com.write(f"D{position}")  # Set position
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

        await self.com.write(f"GAINS({gains})")
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

        await self.com.write(f"V{velocity}")
        response = await self.command("V")
        response = float(response.strip()[1:])

        return response == velocity
