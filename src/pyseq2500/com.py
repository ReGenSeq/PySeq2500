from pyseq_core.base_com import BaseCOM
from serial import Serial
from attrs import define, field
from functools import cached_property
import logging
import io
from typing import Union

LOGGER = logging.getLogger("PySeq")


@define(kw_only=True)
class SerialCOM(BaseCOM):
    tx: Serial = field(init=False)
    rx: Serial = field(init=False)
    rx_address: str = field(default=None)
    com: io.TextIOWrapper = field(init=False)
    _connected: bool = field(default=False)

    @cached_property
    def prefix(self):
        return self.config["prefix"]

    @cached_property
    def suffix(self):
        return self.config["suffix"]

    async def connect(self, baudrate: int = 9600, timeout: int = 1) -> Union[None, str]:
        if not self._connected:
            async with self.lock:
                tx = Serial(port=self.address, baudrate=baudrate, timeout=timeout)
                address = self.address

                if self.rx_address is not None:
                    # Add seperate response serial port, like for HiSeq 2500 FPGA
                    rx = Serial(
                        port=self.rx_address, baudrate=baudrate, timeout=timeout
                    )
                    address += " and {self.rx._address}"
                else:
                    # use the same serial port for responses, most instrumentation
                    rx = tx

                self.com = io.TextIOWrapper(
                    io.BufferedRWPair(tx, rx), encoding="ascii", errors="ignore"
                )
                self._connected = True

            return f"{self.name} connected to {address}"

    async def write(self, command: str) -> str:
        cmdid = self.bump_cmdid()
        command = f"{self.prefix}{command}{self.suffix}"
        self.com.write(command)
        self.com.flush()
        LOGGER.debug(f"{self.name} :: tx {cmdid} :: {command}")
        return cmdid

    async def read(self, cmdid) -> str:
        response = self.com.readline()
        LOGGER.debug(f"{self.name} :: rx {cmdid}:: {response}")
        return response

    async def command(self, command: str) -> str:
        async with self.lock:
            await self.write(command)
            return await self.read()

    async def close(self):
        async with self.lock:
            self.tx.close()
            if self.rx_address is not None:
                self.rx.close()
            self._connected = False


@define(kw_only=True)
class EmulatedSerialCOM(BaseCOM):
    @cached_property
    def prefix(self):
        return self.config["prefix"]

    @cached_property
    def suffix(self):
        return self.config["suffix"]

    async def connect(self) -> None:
        """Emulate connection to serial port"""
        async with self.lock:
            LOGGER.debug(f"{self.name} emulating connection to {self.address}")
            self.com = True

    async def close(self) -> bool:
        """Emulate closing a connection to serial port.

        Returns:
            bool: True if the connection is gracefully closed, otherwise False.
        """
        async with self.lock:
            LOGGER.debug(f"{self.name} emulating closing connection to {self.address}")
        return True

    def response(self, response):
        """Format response"""
        return f"{self.prefix}{response}{self.suffix}"
