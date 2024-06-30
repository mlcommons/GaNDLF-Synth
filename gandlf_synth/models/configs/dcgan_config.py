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
            len(model_config["input_shape"]) == model_config["n_dimensions"]
        ), " `input_shape` parameter in model config needs to have number of elements equalt to `n_dimensions`."

        for dim in model_config["input_shape"]:
            assert dim > 0 and isinstance(
                dim, int
            ), "All elements of `input_shape` must be positive integers."
        if "leaky_relu_slope" in model_config["architecture"]:
            assert (
                model_config["architecture"]["leaky_relu_slope"] > 0
            ), "Leaky ReLU slope must be greater than 0."

    def _set_default_params(self, model_config: dict) -> dict:
        for key, value in self.ARCHITECTURE_DEFAULT_PARAMS.items():
            if key not in model_config["architecture"]:
                model_config["architecture"][key] = value
                warn(
                    f"Parameter {key} not found in the `architecture` field of `model_config`. Setting value to default: {value}.",
                    UserWarning,
                )
        # special case for the DCGAN model
        model_config["output_shape"] = model_config.get(
            "output_shape", model_config["tensor_shape"]
        )
        warn(
            "Parameter `output_shape` not found in the `model_config`. Setting value to `tensor_shape`.",
            UserWarning,
        )

        return model_config
