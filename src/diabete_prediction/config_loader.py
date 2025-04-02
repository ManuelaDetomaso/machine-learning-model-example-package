import configparser
import json
import os
from importlib import resources  # Python 3.9+

from diabete_prediction.utils import parse_value

import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> json:
    """Function to load a config.ini parsing it as a json file

    Args:
        config_path (str, optional): the path to a config.ini,
        for instance 'src/diabete_prediction/config.ini'. Defaults to None.

    Returns:
        json: parsed config.ini as json file
    """
    config = configparser.ConfigParser()

    if config_path:
        logger.info("🔧 Loading config from path: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config.read_file(f)
    else:
        try:
            # Try to read it from installed package (wheel or site-packages)
            with (
                resources.files("diabete_prediction")
                .joinpath("config.ini")
                .open("r", encoding="utf-8") as f
            ):
                config.read_file(f)
        except FileNotFoundError:
            # Fallback: detect editable mode and read from src/
            dev_path = os.path.join("src", "diabete_prediction", "config.ini")
            logger.warning("🛠 Fallback to dev config path: %s", dev_path)
            with open(dev_path, "r", encoding="utf-8") as f:
                config.read_file(f)

    parsed_config = {
        section: {k: parse_value(v) for k, v in config[section].items()}
        for section in config.sections()
    }
    return parsed_config
