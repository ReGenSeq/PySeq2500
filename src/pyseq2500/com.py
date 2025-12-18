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

LOGGER = logging.getLogger("PySeq")


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

    async def connect(self, baudrate: int = 9600, timeout: int = 1) -> Union[None, str]:
        if not self._connected:
            async with self.lock:
                self.tx = Serial(port=self.address, baudrate=baudrate, timeout=timeout)
                address = self.address

                if self.rx_address is not None:
                    # Add seperate response serial port, like for HiSeq 2500 FPGA
                    self.rx = Serial(
                        port=self.rx_address, baudrate=baudrate, timeout=timeout
                    )
                    address += " and {self.rx._address}"
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

    async def read(self, cmdid: str) -> str:
        response = self.com.readline()
        LOGGER.debug(f"{self.name} :: rx {cmdid} :: {response}")
        return response

    async def command(self, command: str) -> str:
        async with self.lock:
            cmdid = await self.write(command)
            return await self.read(cmdid)

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

    def response(self, response):
        """Format response"""
        return f"{self.prefix}{response}{self.suffix}"


address_dict = {dev.serial_number: dev.device for dev in comports()}
COM_DICT = map_coms(SerialCOM, address_dict, HW_CONFIG)
"""Dictionary mapping address (dev.serial_number) to COM port (dev.device)"""
