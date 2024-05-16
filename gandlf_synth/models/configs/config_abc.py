from abc import ABC, abstractmethod


class AbstractModelConfig(ABC):
    """
    Class representing abstraction for model configuration. For any new
    models, we would require to inherit this class and implement the custom config object
    for the new architecture.
    """

    def __init__(self, model_config: dict) -> None:
        super().__init__()
        self._validatie_params(model_config)
        config = self._set_default_params(model_config)
        self._create_properites_from_config(config)

    @staticmethod
    @abstractmethod
    def _validatie_params(model_config: dict) -> None:
        """
        This method checks if the input model configuration is valid by
        checking if the required keys are present in the dictionary.
        Args:
            model_config (dict): The model configuration dictionary.
        """
        pass

    @staticmethod
    @abstractmethod
    def _set_default_params(model_config: dict) -> dict:
        """
        This method sets the default parameters for the model configuration
        if the user has not provided any.
        """
        pass

    @classmethod
    def _create_properites_from_config(cls, model_config: dict) -> None:
        """
        This method creates the properties from the model configuration
        dictionary.
        Args:
            model_config (dict): The model configuration dictionary.
        """
        for key, value in model_config.items():
            setattr(cls, key, value)


# Just an example of configuration class
class TestModelConfig(AbstractModelConfig):
    """
    Test model configuration class.
    """

    @staticmethod
    def _validatie_params(model_config: dict) -> None:
        """
        This method checks if the input model configuration is valid by
        checking if the required keys are present in the dictionary.
        Args:
            model_config (dict): The model configuration dictionary.
        """
        assert "test_key" in model_config, "test_key not found in model configuration."

    @staticmethod
    def _set_default_params(model_config: dict) -> dict:
        """
        This method sets the default parameters for the model configuration
        if the user has not provided any.
        """
        if "test_key_opt" not in model_config:
            model_config["test_key_opt"] = "test_value_inserted_by_default"
        return model_config


if __name__ == "__main__":
    test_dict = {"test_key": "test_value"}
    test_model_config = TestModelConfig(test_dict)
    print(test_model_config.test_key)
    print(test_model_config.test_key_opt)
