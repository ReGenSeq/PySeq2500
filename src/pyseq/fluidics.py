from pyseq_core.base_instruments import BasePump, BaseValve
from typing import Union
import logging

LOGGER = logging.getLogger("PySeq")


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


    Inherited BaseInstrument Methods:
        command(Union[str,dict]) -> Union[str,dict]: Send a command string/dict
            to the pump.
    """

    async def initialize(self):
        """Initializes the pump hardware.

        This method perfrom any necessary hardware configurations.
        """
        pass

    async def shutdown(self):
        """Shuts down the valve hardware.

        This method should gracefully disconnect from the pump, release
        any resources, and put the pump into a safe, parked, or off state.
        """
        pass

    async def status(self) -> bool:
        """Retrieves the current operational status of the valve.

        This method queries the pump hardware to determine if it is
        connected, responsive, and in a ready state.

        Returns:
            bool: True if the valve is operational, False otherwise.
        """
        pass

    async def configure(self):
        """Configures the pump based on its loaded settings.

        This method applies specific configuration parameters from `self.config`
        to the physical pump hardware. This might include setting operating modes,
        calibration values, or other device-specific settings.
        """
        pass

    async def pump(
        self, volume: Union[float, int], flow_rate: Union[float, int], **kwargs
    ):
        """Pump a specified volume at a specified flow rate from inlet to outlet of flowcell.

        This method should be implemented by subclasses to control the physical
        pump to dispense a given volume of liquid at a particular flow rate.

        Args:
            volume (Union[float, int]): The volume of liquid to pump.
            flow_rate (Union[float, int]): The rate at which to pump the liquid.
            **kwargs: Additional keyword arguments that might be specific to
                      a particular pump implementation (e.g., pause_time, waste_flow_rate).
        Returns:
            bool: True if succesfully pumped volume, otherwise False.
        """
        pass

    async def reverse_pump(
        self, volume: Union[float, int], flow_rate: Union[float, int], **kwargs
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
        pass


class Valve(BaseValve):
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
        pass

    async def shutdown(self):
        """Shuts down the valve hardware.

        This method should gracefully disconnect from the valve, release
        any resources, and put the valve into a safe state.
        """
        pass

    async def status(self) -> bool:
        """Retrieves the current operational status of the valve.

        This method queries the valve hardware to determine if it is
        connected, responsive, and in a ready state.

        Returns:
            bool: True if the valve is operational, False otherwise.
        """
        pass

    async def configure(self):
        """Configures the valve based on its loaded settings.

        This method applies specific configuration parameters from `self.config`
        to the physical valve hardware. This might include setting operating modes,
        calibration values, or other device-specific settings.
        """
        pass

    async def select(self, port: Union[str, int], **kwargs) -> bool:
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
        pass

    async def current_port(self) -> Union[str, int]:
        """Read the current active port from the valve.

        This method should be implemented by subclasses to query the physical
        valve and retrieve its currently selected port.

        Returns:
            Union[str, int]: The identifier of the current active port.
        """
