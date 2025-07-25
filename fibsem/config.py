import os

import yaml

import fibsem

METADATA_VERSION = "v3"

SUPPORTED_COORDINATE_SYSTEMS = [
    "RAW",
    "SPECIMEN",
    "STAGE",
    "Raw",
    "raw",
    "specimen",
    "Specimen",
    "Stage",
    "stage",
]

REFERENCE_HFW_WIDE = 2750e-6
REFERENCE_HFW_LOW = 900e-6
REFERENCE_HFW_MEDIUM = 400e-6
REFERENCE_HFW_HIGH = 150e-6
REFERENCE_HFW_SUPER = 80e-6
REFERENCE_HFW_ULTRA = 50e-6

REFERENCE_RES_SQUARE = [1024, 1024]
REFERENCE_RES_LOW = [768, 512]
REFERENCE_RES_MEDIUM = [1536, 1024]
REFERENCE_RES_HIGH = [3072, 2048]
REFERENCE_RES_SUPER = [6144, 4096]

# standard imaging resolutions
STANDARD_RESOLUTIONS = [
    "384x256",
    "768x512",
    "1536x1024",
    "3072x2048",
    "6144x4096",
]
SQUARE_RESOLUTIONS = [
    "256x256",
    "512x512",
    "1024x1024",
    "2048x2048",
    "4096x4096",
    "8192x8192",
]
STANDARD_RESOLUTIONS_LIST = [
    [int(x) for x in res.split("x")] for res in STANDARD_RESOLUTIONS
]
SQUARE_RESOLUTIONS_LIST = [
    [int(x) for x in res.split("x")] for res in SQUARE_RESOLUTIONS
]
AVAILABLE_RESOLUTIONS = SQUARE_RESOLUTIONS + STANDARD_RESOLUTIONS
DEFAULT_STANDARD_RESOLUTION = "1536x1024"
DEFAULT_SQUARE_RESOLUTION = "1024x1024"

MILL_HFW_THRESHOLD = 0.01  # 1.0% of the image

BASE_PATH = os.path.dirname(
    fibsem.__path__[0]
)  # TODO: figure out a more stable way to do this
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(BASE_PATH, "fibsem", "log", "data")
DATA_ML_PATH: str = os.path.join(BASE_PATH, "fibsem", "log", "data", "ml")
DATA_CC_PATH: str = os.path.join(BASE_PATH, "fibsem", "log", "data", "crosscorrelation")
DATA_TILE_PATH: str = os.path.join(DATA_PATH, "tile")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "microscope-configuration.yaml"
)



os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DATA_ML_PATH, exist_ok=True)
os.makedirs(DATA_CC_PATH, exist_ok=True)
os.makedirs(DATA_TILE_PATH, exist_ok=True)

DATABASE_PATH = os.path.join(BASE_PATH, "fibsem", "db", "fibsem.db")
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)



def load_yaml(fname):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        fname (str): The path to the YAML file to be loaded.

    Returns:
        dict: A dictionary containing the contents of the YAML file.

    Raises:
        IOError: If the file cannot be opened or read.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config

AVAILABLE_MANUFACTURERS = ["Thermo", "Tescan", "Demo"]
DEFAULT_MANUFACTURER = "Thermo"
DEFAULT_IP_ADDRESS = "192.168.0.1"
SUPPORTED_PLASMA_GASES = ["Argon", "Oxygen", "Nitrogen", "Xenon"]

def get_default_user_config() -> dict:
    """Return the default configuration."""
    return {
        "name":                           "default-configuration",       # a descriptive name for your configuration 
        "ip_address":                     DEFAULT_IP_ADDRESS,            # the ip address of the microscope PC
        "manufacturer":                   DEFAULT_MANUFACTURER,          # the microscope manufactuer, Thermo, Tescan or Demo                       
        "rotation-reference":             0,                             # the reference rotation value (rotation when loading)  [degrees]
        "shuttle-pre-tilt":               35,                            # the pre-tilt of the shuttle                           [degrees]
        "electron-beam-eucentric-height": 7.0e-3,                        # the eucentric height of the electron beam             [metres]
        "ion-beam-eucentric-height":      16.5e-3,                       # the eucentric height of the ion beam                  [metres]
    }


# user configurations -> move to fibsem.db eventually
DEFAULT_USER_CONFIGURATION_YAML: dict = {
    "configurations": {"default-configuration": {"path": None}},
    "default": "default-configuration",
}
USER_CONFIGURATIONS_PATH = os.path.join(CONFIG_PATH, "user-configurations.yaml")
if os.path.exists(USER_CONFIGURATIONS_PATH):
    USER_CONFIGURATIONS_YAML = load_yaml(USER_CONFIGURATIONS_PATH)
else:
    USER_CONFIGURATIONS_YAML = DEFAULT_USER_CONFIGURATION_YAML
USER_CONFIGURATIONS = USER_CONFIGURATIONS_YAML["configurations"]
DEFAULT_CONFIGURATION_NAME = USER_CONFIGURATIONS_YAML["default"]
DEFAULT_CONFIGURATION_PATH = USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME]["path"]


if DEFAULT_CONFIGURATION_PATH is None:
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH
        
print(f"Default configuration {DEFAULT_CONFIGURATION_NAME}. Configuration Path: {DEFAULT_CONFIGURATION_PATH}")

def add_configuration(configuration_name: str, path: str):
    """Add a new configuration to the user configurations file."""
    if configuration_name in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' already exists.")

    USER_CONFIGURATIONS[configuration_name] = {"path": path}
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def remove_configuration(configuration_name: str):
    """Remove a configuration from the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    del USER_CONFIGURATIONS[configuration_name]
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def set_default_configuration(configuration_name: str):
    """Set the default configuration in the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    USER_CONFIGURATIONS_YAML["default"] = configuration_name
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


# default configuration values
DEFAULT_CONFIGURATION_VALUES = {
    "Thermo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
    "Tescan": {
        "ion-column-tilt": 55,
        "electron-column-tilt": 0,
    },
    "Demo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
}


# machine learning
HUGGINFACE_REPO = "patrickcleeve/autolamella"
DEFAULT_CHECKPOINT = "autolamella-mega-20240107.pt"

# feature flags
APPLY_CONFIGURATION_ENABLED = True

# tescan manipulator

TESCAN_MANIPULATOR_CALIBRATION_PATH = os.path.join(CONFIG_PATH, "tescan_manipulator.yaml")

def load_tescan_manipulator_calibration() -> dict:
    """Load the tescan manipulator calibration"""
    from fibsem.utils import load_yaml
    config = load_yaml(TESCAN_MANIPULATOR_CALIBRATION_PATH)
    return config

def save_tescan_manipulator_calibration(config: dict) -> None:
    """Save the tescan manipulator calibration"""
    from fibsem.utils import save_yaml
    save_yaml(TESCAN_MANIPULATOR_CALIBRATION_PATH, config)
    return None