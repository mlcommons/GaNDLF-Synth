from warnings import warn
from copy import deepcopy
from abc import ABC, abstractmethod


class AbstractModelConfig(ABC):
    """
    Class representing abstraction for model configuration. For any new
    models, we would require to inherit this class and implement the custom config object
    for the new architecture.
    """

    def __init__(self, model_config: dict) -> None:
        super().__init__()
        self.model_specifc_default_params = self._prepare_default_model_params()
        self.architecture_default_params = self._prepare_default_architecture_params()
        config = deepcopy(model_config)
        config = self._set_default_params(config)
        config = self._set_default_architecture_params(config)
        self._validate_params(config)
        self._create_properites_from_config(config)

    @staticmethod
    @abstractmethod
    def _validate_params(model_config: dict) -> None:
        """
        This method checks if the input model configuration is valid by
        checking if the required keys are present in the dictionary.
        Args:
            model_config (dict): The model configuration dictionary.
        """
        pass

    @staticmethod
    @abstractmethod
    def _prepare_default_model_params() -> dict:
        """
        This method prepares the default model parameters for the model configuration.
        """
        pass

    @staticmethod
    @abstractmethod
    def _prepare_default_architecture_params() -> dict:
        """
        This method prepares the default architecture parameters for the model configuration.
        """
        pass

    def _set_default_params(self, model_config: dict) -> dict:
        for key, value in self.model_specifc_default_params.items():
            if key not in model_config:
                model_config[key] = value
                warn(
                    f"Parameter {key} not found in the `model_config`. Setting value to default: {value}.",
                    UserWarning,
                )
        return model_config

    def _set_default_architecture_params(self, model_config: dict) -> dict:
        """
        This method sets the default architecture parameters for the model configuration
        if the user has not provided any.
        """
        for key, value in self.architecture_default_params.items():
            if key not in model_config["architecture"]:
                model_config["architecture"][key] = value
                warn(
                    f"Parameter {key} not found in the `architecture` field of `model_config`. Setting value to default: {value}.",
                    UserWarning,
                )
        return model_config

    def _create_properites_from_config(self, model_config: dict) -> None:
        """
        This method creates the properties from the model configuration
        dictionary.
        Args:
            model_config (dict): The model configuration dictionary.
        """
        for key, value in model_config.items():
            setattr(self, key, value)
