from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class UnlabeledStyleGANConfig(AbstractModelConfig):
    """
    Configuration class for the StyleGAN model handling unlabeled data.
    """

    @staticmethod
    def _validate_params(model_config: dict) -> None:
        assert (
            len(model_config["tensor_shape"]) == model_config["n_dimensions"]
        ), " `tensor_shape` parameter in model config needs to have number of elements equalt to `n_dimensions`."

        for dim in model_config["tensor_shape"]:
            assert dim > 0 and isinstance(
                dim, int
            ), "All elements of `tensor_shape` must be positive integers."
        assert len(
            model_config["architecture"]["progressive_layers_scaling_factors"]
        ) == len(
            model_config["architecture"]["progressive_epochs"]
        ), "Number of elements in `progressive_layers_scaling_factors` must be equal to number of elements in `progressive_epochs`."
        assert (
            model_config["default_forward_step"]
            <= len(model_config["architecture"]["progressive_epochs"]) - 1
        ), "Default forward step must be less than or equal to the number of progressive epochs - 1."
        assert (
            model_config["architecture"]["progressive_layers_scaling_factors"][0] == 1
        ), "First element of `progressive_layers_scaling_factors` must be 1."

    @staticmethod
    def _prepare_default_model_params() -> dict:
        return {
            "fixed_latent_vector_seed": 42,  # Seed for the fixed latent vector used when generating eval images at the end of training epoch
            "n_fixed_images_batch_size": 1,  # Batch size of images to gernerate at the end of each training epochs
            "n_fixed_images_to_generate": 8,  # How many images to generate at the end of each training epochs
            "save_eval_images_every_n_epochs": -1,  # Save evaluation images every n epochs, < 0 means never
            "default_forward_step": 1,  # Default step to use for forward pass
            "tesnor_shape": [
                1,
                1,
                1,
            ],  # Shape of the input tensor - this is ignored in Stylegan
        }

    @staticmethod
    def _prepare_default_architecture_params() -> dict:
        return {
            "latent_vector_size": 512,
            "intermediate_latent_size": 512,
            "first_conv_channels": 512,
            "alpha": 1e-7,
            "gradient_penalty_weight": 10,
            "critic_squared_loss_weight": 0.001,
            "progressive_size_starting_value": 4,
            "progressive_size_growth_factor": 2,
            "progressive_layers_scaling_factors": [
                1,
                1,
                1,
                1 / 2,
                1 / 4,
                1 / 8,
                1 / 16,
                1 / 32,
            ],
            "progressive_epochs": [30, 30, 30, 30, 30, 30, 30, 30],
        }
