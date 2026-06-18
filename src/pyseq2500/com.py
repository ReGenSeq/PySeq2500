from pyseq_core.base_com import BaseCOM

import logging
import re
import asyncio
from aioserial import AioSerial
from attrs import define, field
from functools import cached_property, wraps
from serial.tools.list_ports import comports
from serial.serialutil import SerialException
from pyseq_core.utils import map_coms
from pyseq2500.utils import HW_CONFIG


LOGGER = logging.getLogger("PySeq")
try:
    LOOP = asyncio.get_running_loop()
except RuntimeError:
    LOOP = None


def validate_and_retry(pattern, max_attempts=4, cleanup_at=3, clear_lines=1):
    """
    Async decorator that retries a query until the response matches a regex.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if "{id}" in pattern:
                # For tilt motors
                formatted_pattern = pattern.format(id=getattr(self, "id", 1))
            else:
                formatted_pattern = pattern

            regex = re.compile(formatted_pattern)

            for attempt in range(1, max_attempts + 1):
                response = await func(self, *args, **kwargs)

                # Check for match (handling None or non-string types safely)
                if response is not None and regex.search(str(response)):
                    return response

                # Trigger the single flush if we've hit the designated attempt
                if attempt == cleanup_at and hasattr(self, "com"):
                    await self.com.clear(clear_lines)

                # If this was our last shot, raise the alarm
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"{self.name}:: {func.__name__} failed pattern "
                        f"match '{pattern}' after {max_attempts} attempts. "
                        f"Last received: '{response}'"
                    )

                await asyncio.sleep(0.1)

        return wrapper

    return decorator


@define(kw_only=True)
class SerialCOM(BaseCOM):
    tx: AioSerial = field(init=False)
    rx: AioSerial = field(init=False)
    rx_address: str = field(default=None)
    # com: io.TextIOWrapper = field(init=False)

    @cached_property
    def prefix(self):
        return self.config["prefix"]

    @cached_property
    def suffix(self):
        return self.config["suffix"]

    async def connect(self, baudrate: int = 0, timeout: int = 0) -> bool:
        if not self._connected:
            if baudrate == 0:
                baudrate = self.config["baudrate"]
            if timeout == 0:
                timeout = self.config["timeout"]

            if self.rx_address is None:
                self.rx_address = self.config.get("rx_address", None)

            async with self.lock:
                try:
                    # self.tx = Serial(port=self.address, baudrate=baudrate, timeout=timeout)
                    address = self.address
                    self.tx = AioSerial(
                        port=self.address, baudrate=baudrate, timeout=timeout, loop=LOOP
                    )
                    self._connected = True

                    if self.rx_address is None:
                        LOGGER.debug(f"{self.name} connected to {address}")

                except SerialException as e:
                    LOGGER.error(e)
                    LOGGER.error(f"{self.name} could not connect to {address}")

                if self.rx_address is not None:
                    # Add seperate response serial port, like for HiSeq 2500 FPGA
                    port = address_dict.get(self.rx_address, None)
                    if port is not None:
                        try:
                            self.rx = AioSerial(
                                port=port, baudrate=baudrate, timeout=timeout, loop=LOOP
                            )
                            self.rx_address = port
                            address += f" and {port}"
                            LOGGER.debug(f"{self.name} connected to {address}")
                        except SerialException as e:
                            LOGGER.error(e)
                            LOGGER.error(f"{self.name} could not connect to {address}")
                            self._connected = False
                    else:
                        LOGGER.error(f"Could not find COM port {port} for {self.name}")
                else:
                    # use the same serial port for responses, most instrumentation
                    self.rx = self.tx

        return self._connected

    async def write(self, command: str) -> str:
        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        await self.tx.write_async(command.encode())
        # self.com.flush()
        LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")
        return cmdid

    async def read(self, timeout: float = 10) -> str:
        cmdid = f"{self._cmdid:04d}"
        response = ""
        try:
            async with asyncio.timeout(timeout):
                while len(response) == 0:
                    response = await self.rx.readline_async(size=-1)
                    response = response.decode(errors="ignore").strip()
                    if len(response) == 0:
                        await asyncio.sleep(0.5)
                LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
                return response
        except asyncio.TimeoutError:
            LOGGER.warning(f"{self.name} :: rx {cmdid} :: response timed out")
            return response

    async def clear(self, lines: int = 1):
        LOGGER.warning(f"{self.name} :: clearing responses")
        for i in range(lines):
            response = await self.rx.readline_async(size=-1)
            LOGGER.debug(f"{self.name} :: rx :: {response}")
        while len(response) > 0:
            response = await self.rx.readline_async(size=-1)
            LOGGER.debug(f"{self.name} :: rx :: {response}")

    async def command(self, command: str, read=True, delay=0.1, timeout=10) -> str:
        async with self.lock:
            await self.write(command)

            if read:
                await asyncio.sleep(delay)

                if isinstance(read, bool):
                    read = 1
                for _ in range(read):
                    response = await self.read(timeout=timeout)
                return response  # pyright: ignore[reportPossiblyUnboundVariable]
            else:
                return ""

    async def close(self):
        async with self.lock:
            self.tx.close()
            if self.rx_address is not None:
                self.rx.close()
            self._connected = False


@define(kw_only=True)
class EmulatedSerialCOM(BaseCOM):
    _connect: bool = field(default=False)

    @property
    def connected(self):
        return self._connected

    @cached_property
    def prefix(self):
        return self.config["prefix"]

    @cached_property
    def suffix(self):
        return self.config["suffix"]

    async def connect(self) -> bool:
        """Emulate connection to serial port"""
        if not self._connected:
            async with self.lock:
                self.com = True
                self._connected = True
            LOGGER.debug(f"{self.name} connected to {self.address}")
        return self._connected

    async def close(self) -> bool:
        """Emulate closing a connection to serial port.

        Returns:
            bool: True if the connection is gracefully closed, otherwise False.
        """
        async with self.lock:
            LOGGER.debug(f"{self.name} emulating closing connection to {self.address}")
            self._connected = False
        return True

    async def write(self, command: str) -> str:
        cmdid = self.bump_cmdid()
        LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")
        return cmdid

    async def read(self, timeout=None):
        pass

    def response(self, response) -> str:
        """Format response"""
        return f"{self.prefix}{response}{self.suffix}"


address_dict = {dev.serial_number: dev.device for dev in comports()}
COM_DICT = map_coms(SerialCOM, address_dict, HW_CONFIG)  # pyright: ignore[reportArgumentType]
"""Dictionary mapping address (dev.serial_number) to COM port (dev.device)"""
