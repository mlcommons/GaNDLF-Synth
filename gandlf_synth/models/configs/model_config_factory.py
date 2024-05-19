from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.configs.dcgan_config import UnlabeledDCGANConfig

from typing import Type


class ModelConfigFactory:
    AVAILABLE_MODEL_CONFIGS = {"unlabeled_dcgan": UnlabeledDCGANConfig}

    @staticmethod
    def _parse_config_name(parameters: dict) -> str:
        """
        Method that parses the data from config file to get the model config name.

        Args:
            parameters (dict): Dictionary containing the parameters from the config file.

         Returns:
            str: Name of the model configuration.
        """
        model_name = parameters["model_config"]["model_name"]
        labeling_paradigm = parameters["model_config"]["labeling_paradigm"]
        return f"{labeling_paradigm}_{model_name}"

    def get_config(self, parameters: str) -> Type[AbstractModelConfig]:
        """
        Factory function to create a model configuration based on the config name.

        Args:
            parameters (dict): Dictionary containing the parameters from the config file.
        Returns:
            model_config: A model configuration object based on the config name.
        """
        config_name = self._parse_config_name(parameters)
        assert config_name in self.AVAILABLE_MODEL_CONFIGS.keys(), (
            f"Model configuration {config_name} not found. "
            f"Available configurations: {self.AVAILABLE_MODEL_CONFIGS.keys()}"
        )
        return self.AVAILABLE_MODEL_CONFIGS[config_name]
