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
        # for dcgan, no specific parameters related to architecutre are required
        # so we can just pass. For other archs this will probably be different
        # we can also think if in such validatiors we will perform specific checks,
        # such as leaky_relu_slope > 0 and <1, etc.
        pass

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
