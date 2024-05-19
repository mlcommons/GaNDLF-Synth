from warnings import warn
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


# TODO: We need to think if we want to create separate classes for
# different labeling strategies or if we want to have a single class
# that handles all of them. The latter would limit the number of classes
# we need to create, but would make the code more complex.
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
            "generator_output_size" in model_config["architecture"]
        ), "`generator_output_size` must be defined in the `architecture` field of `model_config`."
        assert (
            "discriminator_input_size" in model_config["architecture"]
        ), "`discriminator_input_size` must be defined in the `architecture` field of `model_config`."

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
