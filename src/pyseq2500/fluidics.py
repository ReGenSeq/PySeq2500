from pyseq_core.base_instruments import BasePump, BaseValve
from pyseq_core.utils import parse, DEFAULT_CONFIG
from typing import Union
import logging
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
import re
import asyncio
from functools import cached_property
from warnings import warn


LOGGER = logging.getLogger("PySeq")

# /1`0 or /0?100
# /1 or /0 indicates pump id
# ` indicates pump ready, ? indicates pump busy
# 0 or 100 indicates pump step position
PUMP_STATUS = re.compile(r"(\d+)(`|\@)(\d+)")


def pump_status_converter(ready: str) -> bool:
    if ready == "`":
        return True
    elif ready == "@":
        return False
    else:
        raise ValueError(f"Invalid ready code: {ready}")


@define
class PumpStatus:
    id: int = field()
    ready: bool = field(converter=pump_status_converter)
    position: int = field(converter=int)


@define(kw_only=True)
class Pump(BasePump):
    """Concrete implementation of a pump instrument.

    This class provides a specific implementation for the abstract methods
    defined in `BasePump`, allowing for control of a physical pump device.
    It handles communication with the pump hardware to pump a set volume at
    a set flow rate and read its current state.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific pump instance.
        com (BaseCOM): The communication interface configured for this pump.
        config (dict): The loaded configuration settings for this pump.

    Inherited BasePump Attributes:
        min_volume (Union[float, int]): The minimum allowed volume.
        max_volume (Union[float, int]): The maximum allowed volume.
        min_flow_rate (Union[float, int]): The minimum allowed flow rate.
        max_flow_rate (Union[float, int]): The maximum allowed flow rate.

    Pump Attributes:
        _interval (Union[int, float]): Time (s) between status queries.
        _ready (bool): Cached ready status of the pump.
        _position (int): Cached position of the pump piston.
        barrels_per_lane (int): Number of syringe barrels dedicated to each flow cell lane.

    Inherited BaseInstrument Methods:
        command(Union[str,dict]) -> Union[str,dict]: Send a command string/dict
            to the pump.
    """

    _interval: Union[int, float] = field(default=1)
    _ready: bool = field(init=False)
    _position: int = field(init=False)
    barrels_per_lane = field(init=False)

    @cached_property
    def barrel_volume(self):
        return self.config["barrel_volume"]

    @cached_property
    def steps(self):
        return self.config["steps"]

    @cached_property
    def min_sps(self):
        return self.config["min_sps"]

    @cached_property
    def max_sps(self):
        return self.config["max_sps"]

    async def initialize(self):
        """Initializes the pump and configures volume and flow rate limits."""
        await self.command("W4R")  # initialize/reset
        await self.command("&")  # firmware version
        await self.command("#")  # checksum
        await self.wait_for_ready()

    async def shutdown(self):
        """Return pump to idle ready state."""

        await self.wait_for_ready()
        await self.command("A0V7000OR")  # push all fluid out to waste
        await self.wait_for_ready()
        await self.command("A0IR")  # move valve to default state
        await self.wait_for_ready()

    async def status(self) -> bool:
        """Query pump for status and position.

        Returns:
            bool: True if the pump is ready, False otherwise.
        """
        response = await self.command("?")
        response = parse(PUMP_STATUS, response, PumpStatus)

        self.ready = response.ready
        self.position = response.position

        return self.ready

    async def wait_for_ready(self, delay: Union[int, float] = 0):
        """Query and wait for pump to be in ready state.

        Use the delay argument to avoid unneccesarily querying the pump,
        especially after moving the piston.

        Args:
            delay (Union[int, float]): The target position to move the stage to.

        """
        if delay > 0:
            await asyncio.sleep(delay)
        while not await self.status():
            await asyncio.sleep(self._interval)

    async def configure(self):
        """Configures volume and flow rate limits based on barrels_per_lane.

        The number of syringe barrels dedicated to a flow cell lane is set by
        the `barrels_per_lane` setting in the pump section of the machine_settings,
        and is used to calculate the min/max volume and min/max flow rate per lane.
        """

        self.barrels_per_lane = self.config["barrels_per_lane"]

        self.max_volume = self.barrel_volume * self.barrels_per_lane
        self.min_volume = self.max_volume / self.steps
        self.max_flow_rate = int(self.max_sps * self.min_volume * 60)
        self.min_flow_rate = int(self.min_sps * self.min_volume * 60)

        # Update hardware settings
        self.config["volume"]["min_val"] = self.min_volume
        self.config["volume"]["max_val"] = self.max_volume
        self.config["flow_rate"]["min_val"] = self.min_flow_rate
        self.config["flow_rate"]["max_val"] = self.max_flow_rate

        units = self.config["volume"]["units"]
        LOGGER.debug(f"{self.name}: min volume = {self.min_volume} {units}")
        LOGGER.debug(f"{self.name}: max volume = {self.max_volume} {units}")
        units = self.config["flow_rate"]["units"]
        LOGGER.debug(f"{self.name}: min flow rate = {self.min_flow_rate} {units}")
        LOGGER.debug(f"{self.name}: max flow rate = {self.max_flow_rate} {units}")

    def vol_to_step(self, volume: Union[int, float]) -> int:
        units = self.config["volume"]["units"]
        if volume > self.max_volume:
            warn(f"{volume} > max volume, only pumping {self.max_volume} {units}")
            return self.max_volume
        if volume < self.min_volume:
            warn(f"{volume} < min volume, pumping {self.min_volume} {units}")
            return self.min_volume
        else:
            return int(round(volume / self.max_volume * self.steps))

    def flow_to_sps(self, flow: Union[int, float]) -> int:
        units = self.config["flow_rate"]["units"]

        if flow > self.max_flow_rate:
            warn(f"{flow} > max flow rate, pumping at {self.max_flow_rate} {units}")
            flow = self.max_flow_rate

        if flow < self.min_flow_rate:
            warn(f"{flow} < min flow rate, pumping at {self.min_flow_rate} {units}")
            flow = self.min_flow_rate

        return int(round(flow / 60 * self.steps / self.max_volume))

    async def pump(
        self,
        volume: Union[float, int],
        flow_rate: Union[float, int],
        pause_time: Union[float, int] = 0,
        waste_flow_rate: Union[float, int] = 10000,
    ):
        """Pump a specified volume at a specified flow rate from inlet to outlet of flowcell.

        This method should be implemented by subclasses to control the physical
        pump to dispense a given volume of liquid at a particular flow rate.

        Args:
            volume (Union[float, int]): The volume (uL) of liquid to pump.
            flow_rate (Union[float, int]): The rate (uL/min) at which to pull in from the flow cell.
            pause_time (Union[float, int]): Time (s) to wait between aspirating and dispensing
            waste_flow_rate (Union[float, int]): The rate (uL/min) at which to push out to waste.
            **kwargs: Additional keyword arguments that might be specific to
                      a particular pump implementation (e.g., pause_time, waste_flow_rate).
        Returns:
            bool: True if succesfully pumped volume, otherwise False.
        """
        await self.wait_for_ready()

        # Aspirate from flowcell
        pos = self.vol_to_step(volume)
        sps = self.flow_to_sps(flow_rate)
        delay = volume / flow_rate * 60 - 2
        while self.position != pos:
            await self.command(f"IA{pos}V{sps}R")
            await self.wait_for_ready(delay=delay)

        # Allow pressure to equalize
        if pause_time is None:
            pause_time = DEFAULT_CONFIG["pump"]["pause_time"]
        await asyncio.sleep(pause_time)

        # Dispense to waste
        if waste_flow_rate is None:
            waste_flow_rate = DEFAULT_CONFIG["pump"]["waste_flow_rate"]
        delay = volume / waste_flow_rate * 60 - 2
        sps = self.flow_to_sps(waste_flow_rate)
        while self.position != 0:
            await self.command(f"OA0V{sps}R")
            await self.wait_for_ready(delay=delay)

        # Move valve to idle input position
        await self.command("IR")
        await self.wait_for_ready()

    async def reverse_pump(
        self,
        volume: Union[float, int],
        flow_rate: Union[float, int],
        pause_time: Union[float, int] = None,
        waste_flow_rate: Union[float, int] = None,
    ):
        """Pump a specified volume at a specified flow rate from outlet to inlet of flowcell.

        This method should be implemented by subclasses to control the physical
        pump to withdraw a given volume of liquid at a particular flow rate.

        Args:
            volume (Union[float, int]): The volume of liquid to reverse pump.
            flow_rate (Union[float, int]): The rate at which to reverse pump the liquid.
            **kwargs: Additional keyword arguments that might be specific to
                      a particular pump implementation.
        Returns:
            bool: True if succesfully pumped volume, otherwise False.
        """

        await self.wait_for_ready()

        # Aspirate from waste
        if waste_flow_rate is None:
            waste_flow_rate = DEFAULT_CONFIG["pump"]["waste_flow_rate"]
        pos = self.vol_to_step(volume)
        sps = self.flow_to_sps(waste_flow_rate)
        delay = volume / waste_flow_rate * 60 - 2
        while self.position != pos:
            await self.command(f"OA{pos}V{sps}R")
            await self.wait_for_ready(delay=delay)

        # Allow pressure to equalize
        if pause_time is None:
            pause_time = DEFAULT_CONFIG["pump"]["pause_time"]
        await asyncio.sleep(pause_time)

        # Dispense to flow cell
        delay = volume / flow_rate * 60 - 2
        sps = self.flow_to_sps(flow_rate)
        while self.position != 0:
            await self.command(f"IA0V{sps}R")
            await self.wait_for_ready(delay=delay)

    @property
    def ready(self) -> bool:
        return self._ready

    @ready.setter
    def ready(self, status):
        self._ready = status

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, status):
        self._position = status


@define(kw_only=True)
class EmulatedPump(EmulatedSerialCOM):
    # id: int = field(default=1)
    valve: str = field(default="I")
    position: int = field(default=0)
    ready: str = field(default="`")
    move_pattern: re.Pattern = field(default=re.compile(r"(I|O)A(\d+)"))
    valve_pattern: re.Pattern = field(default=re.compile(r"(I|O)"))
    query_pattern: re.Pattern = field(default=re.compile(r"\?"))
    init_pattern: re.Pattern = field(default=re.compile(r"W4R"))
    counter: int = field(default=0)
    counter_thresh: int = field(default=2)

    async def command(self, command: str, read: bool = True) -> str:
        """
        Asynchronously emulate sending commands and receiving response from pump.

        Args:
            command (str): The command string to be sent.
        """
        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")

            move_match = re.search(self.move_pattern, command)
            valve_match = re.search(self.valve_pattern, command)
            query_match = re.search(self.query_pattern, command)
            init_match = re.search(self.init_pattern, command)

            if move_match:
                response = self.move(*move_match.groups())
            elif valve_match:
                response = self.valve_move(*valve_match.groups())
            elif query_match:
                response = self.query()
            elif init_match:
                response = self.init()
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def move(self, valve, position):
        """Move syringe piston and valve."""
        self.valve = valve
        self.position = position
        self.ready = "@"
        self.counter = 0
        self.counter_thresh = 2
        return f"{self.ready}"

    def valve_move(self, valve):
        """Move only valve."""
        self.valve = valve
        self.ready = "@"
        self.counter = 0
        self.counter_thresh = 0
        return f"{self.ready}"

    def query(self):
        """Emulate busy behavior and return status of pump"""
        self.counter += 1
        if self.counter > self.counter_thresh:
            self.ready = "`"
            self.counter = 0
        return f"{self.ready}{self.position}"

    def init(self):
        """Initialize pump."""
        self.ready = "@"
        self.position = "0"
        self.counter = 0
        self.counter_thresh = 1
        return f"{self.ready}"


"""
ID -> query id (int) of valve
NP -> query number of ports (int) on valve
CP -> query current position (int) of valve
"""

VALVE_ID = re.compile(r"ID\s+=\s+([\w|\s]+)")
VALVE_NP = re.compile(r"NP\s+=\s+(\d+)")
VALVE_CP = re.compile(r"Position\s+is\s+=\s+(\d+)")


@define(kw_only=True)
class Valve(BaseValve):
    _port: int = field(init=False, converter=int)
    n_ports: int = field(init=False, converter=int)
    _status: bool = field(init=False)
    """Concrete implementation of a valve instrument.

    This class provides a specific implementation for the abstract methods
    defined in `BaseValve`, allowing for control of a physical valve device.
    It handles communication with the valve hardware to select ports and
    read its current state.

    Inherited BaseInstrument Attributes:
        name (str): The name of this specific valve instance.
        com (BaseCOM): The communication interface configured for this valve.
        config (dict): The loaded configuration settings for this valve.

    Inherited BaseValve Attributes:
        _port (Union[str, int]): The cached current port of the valve.
            This attribute is not initialized directly but is set by the `port` setter
            or `initial_port_value` default.
        ports (list): list of valid ports supported by the valve


    Inherited BaseInstrument Methods:
        command(Union[str,dict]) -> Union[str,dict]: Send a command string/dict
            to the valve.

    Inherited BaseValve Properties:
        port (Union[str, int]): get and set current cached port of the valve
    """

    async def initialize(self):
        """Initializes the valve hardware.

        This method performs any necessary hardware configurations and sets the
        Valve to the initial_port position.
        """

        # Record valve firmware
        await self.command("VR")

        # Get valve id and update prefix if needed
        response = await self.command("ID")
        match = re.search(VALVE_ID, response)
        if match:
            ID = match.groups()[0].strip()
            if ID.strip() != "not used":
                self.com.prefix = ID

        # Get number of ports on valve
        response = await self.command("NP")
        match = re.search(VALVE_NP, response)
        if match:
            self.n_ports = match.groups()[0]

        # Update current port
        await self.current_port()

    async def shutdown(self):
        """Put the valve in safe port state."""
        await self.select(self.config["safe_port"])

    async def status(self) -> bool:
        return self._status

    async def configure(self):
        """No configuration needed for valve."""
        pass

    async def select(self, port: Union[str, int], timeout=30) -> bool:
        """Select a specific port on the valve.

        This method should be implemented by subclasses to send commands to the
        physical valve to switch to the specified port.

        Args:
            port (Union[str, int]): The identifier of the port to select.
            **kwargs: Additional keyword arguments that might be specific to
                      a particular valve implementation (e.g., speed, timeout).
        Returns:
            bool: True if successfully selected port, otherwise False.
        """

        async with asyncio.timeout(timeout):
            while self.port != port:
                await self.command(f"GO{port}", read=False)
                position = await self.current_port()
                if position != port:
                    self._status = False
        self._status = True

    async def current_port(self) -> int:
        """Read the current active port from the valve.

        This method should be implemented by subclasses to query the physical
        valve and retrieve its currently selected port.

        Returns:
            Union[str, int]: The identifier of the current active port.
        """
        response = await self.command("CP")
        match = re.search(VALVE_CP, response)
        if match:
            self.port = match.groups()[0]
            self._status = True
            return self.port
        else:
            self._status = False


@define(kw_only=True)
class EmulatedValve(EmulatedSerialCOM):
    _id: int = field(default=1)
    n_ports: int = field(init=False)
    position: int = field(default=1)
    go_pattern: re.Pattern = field(default=re.compile(r"GO(\d+)"))
    cp_pattern: re.Pattern = field(default=re.compile(r"CP"))
    id_pattern: re.Pattern = field(default=re.compile(r"ID"))
    np_pattern: re.Pattern = field(default=re.compile(r"NP"))

    @n_ports.default
    def get_n_ports(self):
        match = re.search(re.compile(r"Valve(\d+)"), self.name)
        if match:
            return int(match.groups()[0])
        else:
            return 24

    async def command(self, command: str, read: bool = True) -> str:
        """
        Asynchronously emulate sending commands and receiving response from pump.

        Args:
            command (str): The command string to be sent.
        """
        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")

            go_match = re.search(self.go_pattern, command)
            cp_match = re.search(self.cp_pattern, command)
            id_match = re.search(self.id_pattern, command)
            np_match = re.search(self.np_pattern, command)

            if go_match:
                response = self.go(*go_match.groups())
            elif cp_match:
                response = self.cp()
            elif id_match:
                response = self.id()
            elif np_match:
                response = self.np()
            else:
                LOGGER.debug(f"{self.name}: Unknown command {command} to respond to")
                response = ""

            if read:
                response = f"{self.prefix}{response}{self.suffix}"
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response

    def go(self, position) -> str:
        """Move valve to position."""
        self.position = int(position)
        return ""

    def cp(self) -> str:
        """Return current position of valve."""
        return f"Position is  = {self.position}"

    def id(self) -> str:
        """Return ID of valve."""
        return f"ID = {self._id}"

    def np(self) -> str:
        """Return number of available ports on the valve."""
        return f"NP = {self.n_ports}"
