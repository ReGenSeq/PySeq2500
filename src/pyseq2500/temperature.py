from pyseq_core.base_instruments import BaseTemperatureController
from pyseq2500.com import EmulatedSerialCOM
from attrs import define, field
from functools import cached_property
import re
import asyncio
import logging

LOGGER = logging.getLogger("PySeq")

"""
ARM9 CHEM Board Serial Commands

INIT                        -- initialize board
?IDN                        -- get firmware version
FCTEMP:<ch>:<val>           -- set flowcell TEC setpoint (ch = 0 or 1)
FCTEMP:<ch>:<P|I|D|S|F>:<val> -- set flowcell PID parameter
?FCTEMP:<ch>                -- read flowcell temperature (response: <T>A1)
FCTEC:<ch>:<0|1>            -- turn flowcell TEC off (0) or on (1)
RETEC:<ch>:<P|I|D|S|F>:<val> -- set chiller TEC PID parameter (ch = 0, 1, or 2)
RETEMP:<ch>:<val>           -- set chiller TEC setpoint
?RETEMP:3                   -- read all 3 chiller temperatures (response: <T0>:<T1>:<T2>A1)

PID constants
  Flowcells:   P=0.2, I=0.1, D=0.0, S=1.875, F=6.0
  TEC 0 & 1:  P=0.8, I=0.2, D=0.0, S=1.875, F=6.0
  TEC 2:       P=1.7, I=1.1, D=0.0  (no S or F)
"""

FC_PID: list[tuple[str, float]] = [
    ("P", 0.2),
    ("I", 0.1),
    ("D", 0.0),
    ("S", 1.875),
    ("F", 6.0),
]

TEC_PID: list[tuple[str, float]] = [
    ("P", 0.8),
    ("I", 0.2),
    ("D", 0.0),
    ("S", 1.875),
    ("F", 6.0),
]

TEC2_PID: list[tuple[str, float]] = [
    ("P", 1.7),
    ("I", 1.1),
    ("D", 0.0),
]


@define(kw_only=True)
class EmulatedARM9(EmulatedSerialCOM):
    """Emulated ARM9 CHEM board for testing.

    A single instance is shared across all three temperature controller
    instances in tests, mirroring the production setup where all three
    share one physical SerialCOM.

    State is maintained per-channel so controllers do not interfere with
    each other.
    """

    name: str = field(default="ChillerTemperatureController")

    # Flowcell state: index 0 = channel A, index 1 = channel B
    _fc_setpoint: list[float] = field(factory=lambda: [25.0, 25.0])
    _fc_temp: list[float] = field(factory=lambda: [25.0, 25.0])
    _fc_on: list[bool] = field(factory=lambda: [False, False])

    # Chiller state: index 0/1/2 for TEC blocks
    _chiller_setpoint: list[float] = field(factory=lambda: [4.0, 4.0, 4.0])
    _chiller_temp: list[float] = field(factory=lambda: [4.0, 4.0, 4.0])

    # Anchored regexes — query patterns checked before set patterns to
    # prevent prefix collision between e.g. ?FCTEMP and FCTEMP
    init_pattern: re.Pattern = field(default=re.compile(r"^INIT$"))
    idn_pattern: re.Pattern = field(default=re.compile(r"^\?IDN$"))
    fc_query_pattern: re.Pattern = field(default=re.compile(r"^\?FCTEMP:([01])$"))
    fc_tec_pattern: re.Pattern = field(default=re.compile(r"^FCTEC:([01]):([01])$"))
    fc_set_pattern: re.Pattern = field(default=re.compile(r"^FCTEMP:([01]):([\d.]+)$"))
    fc_pid_pattern: re.Pattern = field(
        default=re.compile(r"^FCTEMP:([01]):[PIDSF]:([\d.]+)$")
    )
    re_query_pattern: re.Pattern = field(default=re.compile(r"^\?RETEMP:3$"))
    re_set_pattern: re.Pattern = field(default=re.compile(r"^RETEMP:([012]):([\d.]+)$"))
    re_pid_pattern: re.Pattern = field(
        default=re.compile(r"^RETEC:([012]):[PIDSF]:([\d.]+)$")
    )

    async def command(self, command: str, read: bool = True, **kwargs) -> str:
        """Emulate sending a command to the ARM9 board and returning a response.

        Args:
            command (str): The command string to send.
            read (bool): Whether to return a response.

        Returns:
            str: The formatted response string, or "" if read is False.
        """
        cmdid = self.bump_cmdid()
        full_command = f"{self.prefix}{command}{self.suffix}".strip()
        async with self.lock:
            LOGGER.debug(f"{self.name} :: tx {cmdid} :: {full_command}")
            response = self._dispatch(full_command)
            if read:
                response = self.response(response)
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response
            return ""

    def _dispatch(self, command: str) -> str:
        if m := re.search(self.idn_pattern, command):  # noqa: F841
            return "ARM9 Emulated v1.0:A1"
        if m := re.search(self.init_pattern, command):  # noqa: F841
            return "A1"
        if m := re.search(self.fc_query_pattern, command):
            return self._fc_query(int(m.group(1)))
        if m := re.search(self.fc_tec_pattern, command):
            return self._fc_tec(int(m.group(1)), int(m.group(2)))
        if m := re.search(self.fc_set_pattern, command):
            return self._fc_set(int(m.group(1)), float(m.group(2)))
        if re.search(self.fc_pid_pattern, command):
            return "A1"
        if re.search(self.re_query_pattern, command):
            return self._re_query()
        if m := re.search(self.re_set_pattern, command):
            return self._re_set(int(m.group(1)), float(m.group(2)))
        if re.search(self.re_pid_pattern, command):
            return "A1"
        LOGGER.debug(f"{self.name}: Unknown command '{command}'")
        return ""

    def _fc_set(self, ch: int, setpoint: float) -> str:
        self._fc_setpoint[ch] = setpoint
        self._fc_temp[ch] = setpoint  # emulator snaps to setpoint immediately
        return "A1"

    def _fc_query(self, ch: int) -> str:
        return f"{self._fc_temp[ch]:.2f}C:A1"

    def _fc_tec(self, ch: int, on: int) -> str:
        self._fc_on[ch] = bool(on)
        return "A1"

    def _re_set(self, ch: int, setpoint: float) -> str:
        self._chiller_setpoint[ch] = setpoint
        self._chiller_temp[ch] = setpoint
        return "A1"

    def _re_query(self) -> str:
        t = self._chiller_temp
        return f"{t[0]:.2f}C:{t[1]:.2f}C:{t[2]:.2f}:A1"


@define(kw_only=True)
class FlowCellTemperatureController(BaseTemperatureController):
    """Temperature controller for a single flowcell TEC channel on the ARM9 board.

    Holds a reference to a shared SerialCOM (or EmulatedARM9) instance.
    Serialization across all three temperature controllers is provided by
    the asyncio.Lock in SerialCOM.command, matching the FPGA shared-COM pattern.

    The flowcell channel index is read from self.config["fc_channel"] so the
    YAML is the single source of truth for channel assignment.

    Inherited BaseInstrument Attributes:
        name (str): Instrument name, e.g. "FlowCellTemperatureControllerA".
        com (BaseCOM): Shared ARM9 serial communication interface.
        config (dict): Loaded configuration from machine_settings.yaml.

    Inherited BaseTemperatureController Attributes:
        _temperature (Union[float, int]): Cached current temperature in °C.
        min_temperature (cached_property): Minimum allowed temperature (°C).
        max_temperature (cached_property): Maximum allowed temperature (°C).
        temperature_tolerance (cached_property): Acceptable deviation from setpoint (°C).

    """

    _fc_temperature: float = field(init=False, default=float("nan"))
    _temperature: float = field(init=False, default=float("nan"))  # type: ignore[assignment]

    @cached_property
    def fc_channel(self) -> int:
        """Flowcell TEC channel index (0 for A, 1 for B), read from config."""
        return self.config["fc_channel"]

    async def initialize(self) -> None:
        """Set flowcell PID parameters for this channel.

        INIT and ?IDN are intentionally omitted — ChillerTemperatureController
        owns board-level initialization and must be initialized first.
        """
        await self._set_pid()

    async def _set_pid(self) -> None:
        for param, val in FC_PID:
            await self.command(f"FCTEMP:{self.fc_channel}:{param}:{val}")

    async def configure(self, exp_config: dict = {}) -> None:
        pass

    async def shutdown(self) -> None:
        """Turn off the flowcell TEC."""
        await self.fc_off()

    async def status(self) -> bool:
        """True if a valid temperature reading is available."""
        temp = await self.get_temperature()
        return temp == temp  # False if NaN

    async def fc_on(self) -> None:
        """Enable the flowcell TEC."""
        await self.command(f"FCTEC:{self.fc_channel}:1")

    async def fc_off(self) -> None:
        """Disable the flowcell TEC."""
        await self.command(f"FCTEC:{self.fc_channel}:0")

    async def set_temperature(
        self, temperature: float, timeout: float | None = 0.0
    ) -> None:
        """Set the flowcell temperature setpoint.

        Note: does not automatically enable the TEC. Call fc_on() first.

        Args:
            temperature (float): Target temperature in °C.
            timeout (float | None): Seconds to wait for setpoint to be reached.
                0 = fire-and-forget (default). None = wait indefinitely.

        Raises:
            ValueError: If temperature is outside [min_temperature, max_temperature].
        """
        if not (self.min_temperature <= temperature <= self.max_temperature):
            raise ValueError(
                f"{self.name}: temperature {temperature} °C out of range "
                f"[{self.min_temperature}, {self.max_temperature}]"
            )
        await self.command(f"FCTEMP:{self.fc_channel}:{temperature}")
        if timeout is None or timeout > 0:
            await asyncio.wait_for(self.wait_for_temperature(temperature), timeout)

    async def get_temperature(self) -> float:
        """Read the actual flowcell temperature from the ARM9 board.

        Parses the response format: <T>A1

        Returns:
            float: Current flowcell temperature in °C, or NaN on parse failure.
        """
        response = await self.command(f"?FCTEMP:{self.fc_channel}")
        m = re.search(r"([\d.]+)C:A1", response)
        if m:
            self._temperature = float(m.group(1))
        else:
            LOGGER.warning(f"{self.name}: unexpected ?FCTEMP response: {response!r}")
        return self._temperature


@define(kw_only=True)
class ChillerTemperatureController(BaseTemperatureController):
    """Temperature controller for the 3-block reagent chiller on the ARM9 board.

    Owns board-level initialization (INIT, ?IDN) and must be initialized
    before the flowcell temperature controllers.

    Shutdown is a deliberate no-op — the chiller stays cold between runs
    to keep reagents at temperature. Use set_temperature() explicitly if
    a specific parking temperature is needed.

    Inherited BaseInstrument Attributes:
        name (str): "ChillerTemperatureController".
        com (BaseCOM): Shared ARM9 serial communication interface.
        config (dict): Loaded configuration from machine_settings.yaml.

    Inherited BaseTemperatureController Attributes:
        _temperature (Union[float, int]): Cached mean of the three TEC temperatures.
        min_temperature (cached_property): Minimum allowed chiller temperature (°C).
        max_temperature (cached_property): Maximum allowed chiller temperature (°C).
        temperature_tolerance (cached_property): Acceptable deviation from setpoint (°C).

    ChillerTemperatureController Attributes:
        _chiller_temperatures (list[float]): Cached [T0, T1, T2] TEC block temperatures.
    """

    _chiller_temperatures: list[float] = field(
        init=False, factory=lambda: [float("nan")] * 3
    )
    _temperature: float = field(init=False, default=float("nan"))  # type: ignore[assignment]

    async def initialize(self) -> None:
        """Initialize the ARM9 board and set chiller PID parameters.

        Sends INIT and ?IDN on behalf of the whole board. Must be called
        before FlowCellTemperatureController.initialize() on either channel.
        """
        await self.command("INIT")
        await self.command("?IDN")
        await self._set_pid()

    async def _set_pid(self) -> None:
        for ch in range(2):  # TEC blocks 0 and 1
            for param, val in TEC_PID:
                await self.command(f"RETEC:{ch}:{param}:{val}")
        for param, val in TEC2_PID:  # TEC block 2 has no S or F params
            await self.command(f"RETEC:2:{param}:{val}")

    async def configure(self, exp_config: dict = {}) -> None:
        pass

    async def shutdown(self) -> None:
        """No-op: chiller stays running between runs to keep reagents cold."""
        pass

    async def status(self) -> bool:
        """True if all three TEC blocks have valid temperature readings."""
        await self.get_temperature()
        return all(t == t for t in self._chiller_temperatures)  # False if any NaN

    async def set_temperature(
        self,
        temperature: float,
        timeout: float | None = 0.0,
        channel: int | None = None,
    ) -> None:
        """Set the chiller TEC setpoint.

        Args:
            temperature (float): Target temperature in °C.
            timeout (float | None): Seconds to wait for setpoint to be reached.
                0 = fire-and-forget (default). None = wait indefinitely.
            channel (int | None): TEC block 0, 1, or 2. None sets all three.

        Raises:
            ValueError: If temperature is outside [min_temperature, max_temperature].
        """
        if not (self.min_temperature <= temperature <= self.max_temperature):
            raise ValueError(
                f"{self.name}: temperature {temperature} °C out of range "
                f"[{self.min_temperature}, {self.max_temperature}]"
            )
        channels = range(3) if channel is None else [channel]
        for ch in channels:
            await self.command(f"RETEMP:{ch}:{temperature}")
        if timeout is None or timeout > 0:
            await asyncio.wait_for(self.wait_for_temperature(temperature), timeout)

    async def get_temperature(self) -> float:
        """Read all three chiller TEC temperatures from the ARM9 board.

        Parses the response format: <T0>:<T1>:<T2>A1

        Returns:
            float: Mean of the three TEC temperatures in °C, or NaN on parse failure.
        """
        response = await self.command("?RETEMP:3")
        m = re.search(r"([\d.]+)C?:([\d.]+)C?:([\d.]+)C?:A1", response)
        if m:
            self._chiller_temperatures = [float(m.group(i)) for i in range(1, 4)]
            self._temperature = sum(self._chiller_temperatures) / 3
        else:
            LOGGER.warning(f"{self.name}: unexpected ?RETEMP response: {response!r}")
        return self._temperature

    async def wait_for_temperature(
        self, temperature: float, interval: float = 5.0
    ) -> None:
        """Wait until all three TEC blocks are within tolerance of the target.

        Overrides the base class to check all three blocks individually rather
        than comparing against the mean temperature.

        Args:
            temperature (float): Target temperature in °C.
            interval (float): Polling interval in seconds.
        """
        while True:
            await self.get_temperature()
            if all(
                abs(t - temperature) <= self.temperature_tolerance
                for t in self._chiller_temperatures
            ):
                break
            await asyncio.sleep(interval)
