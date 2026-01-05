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

        # Home stage program
        ##        home_program = f"""
        ##        PG 1\r
        ##        HM 1\r
        ##        H\r
        ##        P = {self.home_position}\r
        ##        E\r
        ##        PG\r
        ##        """
        home_program = f"PG 1\rHM 1\rH\rP = {self.home_position}\rE\rPG\r"
        await self.com.write(home_program)

        ##        await self.com.write('PG 1\r')                                        #Enter Program 1
        ##        await self.com.write('HM 1\r')                                        #Homing routine 1
        ##        await self.com.write('H\r')                                           #Hold
        ##        await self.com.write('P = 30000\r')                                   #Set Home posiiton
        ##        await self.com.write('E\r')                                           #End Program
        ##        await self.com.write('PG\r')                                          #Exit Program
        ##        position = await self.get_position()
        ##        LOGGER.info(f"initial position = {position}")
        ##        await self.com.write("EX 1")
        ##        while not await self.status():
        ##            flag = await self.command("PR I1")
        ##            LOGGER.info(f"home flag = {flag}")
        ##        # Rehome stage until I1 reads True
        ##        flag = await self.command("PR I1")
        ##        LOGGER.info(f"home flag = {flag}")
        ##        position = await self.get_position()
        ##        LOGGER.info(f"final position = {position}")
        ##        await self.get_position()
        homed = False
        while not homed:
            # Home stage
            await self.com.write("EX 1")
            while not await self.status():
                # Wait until stage stops moving
                pass
            # Check if home input changes
            await self.move(self.home_position - 1000)
            homed = await self.read_input()
            await self.move(self.home_position)

        ##        while await self.status() and not self.position == self.home_position:
        ##            await self.com.write("EX 1")
        ##            await self.status()
        ##            position = await self.get_position()
        # await self.get_position()
        # self.serial_port.write('PG 1\r')                                        #Start Program 1
        # self.serial_port.write('HM 1\r')                                        #Homing routine 1
        # self.serial_port.write('H\r')                                           #?
        # self.serial_port.write('P = 30000\r')                                   #Set Home posiiton
        # self.serial_port.write('E\r')                                           #?
        # self.serial_port.write('PG\r')                                          #End Program 1
        # self.serial_port.flush()

        # Check if stage is homed correctly
        # cmdid = await self.com.write('EX 1')                                    #Execute home stage program
        # self.position = 30000
        # self.check_position(self.position)
        # homed = self.check_home()
        # if not homed:
        #     self.move(40000)
        #     self.command('EX 1')
        #     self.position = 30000
        #     self.check_position(self.position)
        #     homed = self.check_home()

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
                await self.command(f"MA {position}")
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

        return self.position
