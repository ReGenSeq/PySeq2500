from pyseq_core.base_com import BaseCOM
from serial import Serial
from attrs import define, field
from functools import cached_property
import logging
import io
from typing import Union
from pyseq2500.utils import HW_CONFIG
from serial.tools.list_ports import comports
from pyseq_core.utils import map_coms
import asyncio

LOGGER = logging.getLogger("PySeq")

# TODO: Implement asyncio serial communication
# import serial_asyncio

# class RequestResponseProtocol(asyncio.Protocol):
#     def __init__(self):
#         self.transport = None
#         self.buffer = b""
#         self._waiter = None

#     def connection_made(self, transport):
#         self.transport = transport
#         print("Connected to instrument.")

#     def data_received(self, data):
#         self.buffer += data
#         if b"\r\n" in self.buffer:
#             line, self.buffer = self.buffer.split(b"\r\n", 1)
#             response = line.decode().strip()

#             # If someone is waiting for a response, give it to them
#             if self._waiter and not self._waiter.done():
#                 self._waiter.set_result(response)

#     async def send_command(self, command, timeout=2.0):
#         """Sends a command and waits for the specific response."""
#         if self._waiter and not self._waiter.done():
#             raise RuntimeError("Already waiting for a response!")

#         # Create a 'Future' to represent the upcoming result
#         self._waiter = asyncio.get_running_loop().create_future()

#         # Send the data
#         full_command = (command + "\r\n").encode()
#         self.transport.write(full_command)

#         try:
#             # Wait for data_received to set the result or timeout
#             return await asyncio.wait_for(self._waiter, timeout=timeout)
#         except asyncio.TimeoutError:
#             print(f"Command '{command}' timed out!")
#             return None
#         finally:
#             self._waiter = None

#     def connection_lost(self, exc):
#         if self._waiter and not self._waiter.done():
#             self._waiter.set_exception(ConnectionError("Serial connection lost"))


@define(kw_only=True)
class SerialCOM(BaseCOM):
    tx: Serial = field(init=False)
    rx: Serial = field(init=False)
    rx_address: str = field(default=None)
    com: io.TextIOWrapper = field(init=False)

    @cached_property
    def prefix(self):
        return self.config["prefix"]

    @cached_property
    def suffix(self):
        return self.config["suffix"]

    async def connect(self, baudrate: int = 0, timeout: int = 0) -> Union[None, str]:
        if not self._connected:
            if baudrate == 0:
                baudrate = self.config["baudrate"]
            if timeout == 0:
                timeout = self.config["timeout"]
            async with self.lock:
                self.tx = Serial(port=self.address, baudrate=baudrate, timeout=timeout)
                address = self.address
                if self.rx_address is None:
                    self.rx_address = self.config.get("rx_address", None)

                if self.rx_address is not None:
                    # Add seperate response serial port, like for HiSeq 2500 FPGA
                    port = address_dict.get(self.rx_address, None)
                    if port is not None:
                        self.rx_address = port
                        self.rx = Serial(port=port, baudrate=baudrate, timeout=timeout)
                        address += f" and {port}"
                    else:
                        LOGGER.error(
                            f"Could not find coms with id {port} for {self.name}"
                        )
                else:
                    # use the same serial port for responses, most instrumentation
                    self.rx = self.tx

                self.com = io.TextIOWrapper(
                    io.BufferedRWPair(self.tx, self.rx),
                    encoding="ascii",
                    errors="ignore",
                )
                self._connected = True

            LOGGER.debug(f"{self.name} connected to {address}")

            return self._connected

    async def write(self, command: str) -> str:
        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        self.com.write(command)
        self.com.flush()
        LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")
        return cmdid

    async def read(self, cmdid: str = "") -> str:
        response = self.com.readline()
        if len(cmdid) == 0:
            cmdid = f"{self._cmdid:04d}"
        LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
        return response

    async def command(self, command: str, read=True, delay=0) -> str:
        async with self.lock:
            cmdid = await self.write(command)

            if read:
                await asyncio.sleep(delay)

                if isinstance(read, bool):
                    read = 1
                for _ in range(read):
                    response = await self.read(cmdid)
                return response

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

    async def connect(self) -> None:
        """Emulate connection to serial port"""
        if not self._connected:
            async with self.lock:
                self.com = True
                self._connected = True
            LOGGER.debug(f"{self.name} connected to {self.address}")

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

    async def read(self) -> str:
        pass

    def response(self, response):
        """Format response"""
        return f"{self.prefix}{response}{self.suffix}"


address_dict = {dev.serial_number: dev.device for dev in comports()}
COM_DICT = map_coms(SerialCOM, address_dict, HW_CONFIG)
"""Dictionary mapping address (dev.serial_number) to COM port (dev.device)"""
