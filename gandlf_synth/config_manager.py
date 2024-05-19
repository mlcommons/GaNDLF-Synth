# TODO implement manager that will handle the configs
import yaml
import warnings
from pathlib import Path
from typing import Tuple, Type

from gandlf_synth.parameter_defaults.main_config_defaults import (
    REQUIRED_PARAMETERS,
    BASIC_PARAMETER_DEFAULTS,
)
from gandlf_synth.parameter_defaults.dataloader_defaults import (
    DATALOADER_CONFIG_DEFAULTS,
)
from gandlf_synth.models.configs.available_configs import AVAILABLE_MODEL_CONFIGS

from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class ConfigManager:
    """
    Class responsible for config management.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)

    @staticmethod
    def _read_config(config_path: Path) -> dict:
        """
        Read the configuration file.

        Args:
            config_path (pathlib.Path): The path to the configuration file.

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
        for key, value in BASIC_PARAMETER_DEFAULTS.items():
            if key not in config:
                warnings.warn(
                    f"Parameter {key} not found in the configuration file. Setting value to default: {value}.",
                    UserWarning,
                )
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
        for key, value in DATALOADER_CONFIG_DEFAULTS.items():
            if key not in config:
                warnings.warn(
                    f"Parameter related to dataloader {key} not found in the configuration file. Setting value to default: {value}.",
                    UserWarning,
                )
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

    def prepare_configs(self) -> Tuple[dict, Type[AbstractModelConfig]]:
        """
        Prepare the configuration dictionary and ModelConfig.

        Returns:
            dict: The configuration dictionary.
        """
        config = self._read_config(self.config_path)
        self._validate_general_params_config(config)
        config = self._set_default_params(config)
        config = self._set_dataloader_defaults(config)

        model_name = config["model_config"]["model_name"]
        model_config = AVAILABLE_MODEL_CONFIGS[model_name](config["model_config"])
        config.pop(
            "model_config"
        )  # remove model config from the main config, as it is already stored in the model_config object

        return config, model_config
