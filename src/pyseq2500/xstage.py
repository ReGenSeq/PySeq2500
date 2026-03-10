from pyseq_core.base_instruments import BaseStage
import logging
from pyseq2500.com import EmulatedSerialCOM, SerialCOM
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

    async def command(self, command: str, read: bool = True) -> str:
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

            if read:
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
        self.toggle = 0
        return ""

    def mv(self) -> int:
        """Print 1 if stage is moving, 0 if not moving."""
        self.toggle += 1
        if self.toggle > 1:
            return 0
        else:
            return 1

    def p(self) -> int:
        """Print position of stage."""
        return self.position


@define(kw_only=True)
class XStage(BaseStage):
    """Concrete implementation for the X Stage.

    This class provides a specific implementation for the abstract methods
    defined in `BaseStage`, allowing for control of a physical stage device.
    It handles communication with the stage move and read the stage position.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific pump instance.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BasePump Attributes:
        _position (Union[int, float]): The cached current position of the stage.
            This attribute is not initialized directly but is set by the `position` setter.

    Inherited BaseInstrument Methods:
        command (str) -> str: Send a command string/dict to the stage.
    """

    name: str = field(default="XStage")
    com: SerialCOM = field(default=None)  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def home_position(self):
        return 30000

    async def initialize(self):
        """Initialize and home the XStage."""

        await self.command("\x03")  # Initialize Stage

        # Configure XStage
        await self.command(
            f"EM={self.config.get('echo_mode', 2)}"
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

    async def configure(self, exp_config: dict = {}):
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
                await self.command(f"MA {position}", read=False)
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
