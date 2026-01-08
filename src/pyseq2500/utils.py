import logging
import os
from importlib import resources
import yaml
import shutil
import tomlkit
from pathlib import Path

##from pyseq_core.utils import RESOURCE_PATH
from . import ALIAS


LOGGER = logging.getLogger("PySeq")


# --- MACHINE_SETTINGS Configuration ---
# This section handles the loading of machine-specific hardware configurations.
# Local machine specific settings
RESOURCE_PATH = resources.files(ALIAS)
MACHINE_SETTINGS_PATH = Path.home() / ".config/pyseq/machine_settings.yaml"
MACHINE_SETTINGS_RESOURCE = RESOURCE_PATH.joinpath("resources/machine_settings.yaml")
"""Path to the machine-specific hardware configuration YAML file.

This YAML file stores hardware configurations and settings for all the
instrumentation in the sequencer. Multiple sequencers or versions can be
stored in one file. The top-level key `name` specifies which sequencer or
version to use, and its corresponding settings are loaded into `HW_CONFIG`.

If the file does not exist at `~/.config/pyseq/machine_settings.yaml`,
settings from package resources will be copied and used as a fallback.
"""

if not MACHINE_SETTINGS_PATH.exists():
    # Copy settings from package if local machine setting do not exist
    os.makedirs(MACHINE_SETTINGS_PATH.parent, exist_ok=True)
    os.makedirs(MACHINE_SETTINGS_PATH.parent / "logs", exist_ok=True)
    resource_path = resources.files(ALIAS)
    shutil.copy(MACHINE_SETTINGS_RESOURCE, MACHINE_SETTINGS_PATH)

with open(MACHINE_SETTINGS_PATH, "r") as f:
    all_settings = yaml.safe_load(f)  # Machine config
    machine_name = all_settings["name"]
    HW_CONFIG = all_settings[machine_name]
"""Dictionary containing the hardware configuration for the currently selected machine.

This is loaded from the `MACHINE_SETTINGS_PATH` YAML file, specifically the
section identified by the `name` key in that file.
"""

# --- DEFAULT_CONFIG Configuration ---
# This section handles the loading of default experiment/software configurations.
DEFAULT_CONFIG_PATH = Path.home() / ".config/pyseq/default.toml"
DEFAULT_CONFIG_RESOURCE = RESOURCE_PATH.joinpath("resources/default.toml")
"""Path to the default experiment configuration TOML file.

If `PYTEST_VERSION` environment variable is set and the machine name
contains "test" or "virtual", the default configuration from
package resources is used. Otherwise, it defaults to `~/.config/pyseq/default.toml`.
"""

if not DEFAULT_CONFIG_PATH.exists():
    # Copy settings from package if local machine setting do not exist
    # resource_path = importlib.resources.files(ALIAS)
    shutil.copy(DEFAULT_CONFIG_RESOURCE, DEFAULT_CONFIG_PATH)

# Default settings for experiment/software
if os.environ.get("PYTEST_VERSION") is not None and "test" in machine_name.lower():
    # use default experiment config and machine settings from package resources
    LOGGER.info("Using package default.toml")
    DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_RESOURCE
    # override HW_CONFIG with package resource
    with open(MACHINE_SETTINGS_RESOURCE, "r") as f:
        LOGGER.info("Using package machine_settings.yaml")
        all_settings = yaml.safe_load(f)  # Machine config
        machine_name = all_settings["name"]
        HW_CONFIG = all_settings[machine_name]

# Read default config and machine settings
DEFAULT_CONFIG = tomlkit.parse(open(DEFAULT_CONFIG_PATH).read())
for fc in HW_CONFIG["flowcells"]:
    # Copy barrels_per_lane from HW_CONFIG to DEFAULT_CONFIG to configure pumpes
    p = f"Pump{fc}"
    DEFAULT_CONFIG[p] = {"barrels_per_lane": HW_CONFIG[p]["barrels_per_lane"]}

"""Dictionary containing the default experiment and software configuration.

This is loaded from the `DEFAULT_CONFIG_PATH` TOML file.
"""
