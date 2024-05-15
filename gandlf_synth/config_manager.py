# TODO implement manager that will handle the configs
from typing import Optional, Union
import sys, yaml, ast
import numpy as np
from config.config_defaults import (
    REQUIRED_PARAMETERS,
    PARAMETER_DEFAULTS,
    DATALOADER_CONFIG,
)


class ConfigManager:
    """
    Class responsible for config management.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    @staticmethod
    def _read_config(config_path: str) -> dict:
        """
        Read the configuration file.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            dict: The configuration dictionary.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def _validate_general_params_config(config: dict) -> None:
        """
        Validate if the configuration file contains required options.

        Args:
            config (dict): The configuration dictionary.
        """
        for parameter in REQUIRED_PARAMETERS:
            assert (
                parameter in config
            ), f" Required parameter {parameter} not found in the configuration file."
        # TODO add here more checks, especially check about the model
        # config that always need to be specified, like model name,
        #

    @staticmethod
    def _set_default_params(config: dict) -> dict:
        """
        Set the default parameters for the configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.
        """
        for key, value in PARAMETER_DEFAULTS.items():
            if key not in config:
                config[key] = value
        return config

    @staticmethod
    def _set_dataloader_defaults(config: dict) -> dict:
        """
        Set the default parameters for the dataloader configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.
        """
        for key, value in DATALOADER_CONFIG.items():
            if key not in config:
                config[key] = value
        return config

    # TODO
    @staticmethod
    def _set_preprocessing_defaults(config: dict) -> dict:
        """
        Set the default parameters for the preprocessing configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.
        """
        pass

    # TODO
    @staticmethod
    def _set_augmentation_defaults(config: dict) -> dict:
        """
        Set the default parameters for the augmentation configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.
        """
        pass

    # TODO
    @staticmethod
    def _set_postprocessing_defaults(config: dict) -> dict:
        """
        Set the default parameters for the postprocessing configuration.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            dict: The updated configuration dictionary.
        """
        pass
