from warnings import warn
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class UnlabeledDCGANConfig(AbstractModelConfig):
    """
    Configuration class for the DCGAN model handling unlabeled data.
    """

    ARCHITECTURE_DEFAULT_PARAMS = {
        "latent_vector_size": 100,
        "init_channels_discriminator": 64,
        "init_channels_generator": 512,
        "growth_rate_discriminator": 2,
        "growth_rate_generator": 2,
        "leaky_relu_slope": 0.2,
    }

    @staticmethod
    def _validatie_params(model_config: dict) -> None:
        assert (
            "architecture" in model_config
        ), "Architecture parameters missing in the configuration file. Please scpify the `architecture` field in the `model_config`."
        assert (
            "optimizers" in model_config
        ), "Optimizer parameters missing in the configuration file. Please specify the `optimizers` field in the `model_config`."
        assert (
            "losses" in model_config
        ), "Loss parameters missing in the configuration file. Please specify the `losses` field in the `model_config`."

    @staticmethod
    def _set_default_params(model_config: dict) -> dict:
        for key, value in UnlabeledDCGANConfig.ARCHITECTURE_DEFAULT_PARAMS.items():
            if key not in model_config["architecture"]:
                model_config["architecture"][key] = value
                warn(
                    f"Parameter {key} not found in the `architecture` field of `model_config`. Setting value to default: {value}.",
                    UserWarning,
                )
        return model_config
