from warnings import warn
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class DDPMConfig(AbstractModelConfig):
    """
    Configuartion class for the DDPM model, based on MONAI Generative defaults.

    """

    @staticmethod
    def _prepare_default_model_params() -> dict:
        return {
            "fixed_latent_vector_seed": 42,  # Seed for the fixed latent vector used when generating eval images at the end of training epoch
            "n_fixed_images_batch_size": 1,  # Batch size of images to gernerate at the end of each training epochs
            "n_fixed_images_to_generate": 8,  # How many images to generate at the end of each training epochs
            "save_eval_images_every_n_epochs": -1,  # Save evaluation images every n epochs, < 0 means never
        }

    @staticmethod
    def _prepare_default_architecture_params() -> dict:
        return {
            "num_res_blocks": (2, 2, 2, 2),
            "num_channels": (32, 64, 64, 64),
            "num_train_timesteps": 1000,
            "attention_levels": (False, False, True, True),
            "norm_num_groups": 32,
            "norm_eps": 1e-6,
            "resblock_updown": False,
            "num_head_channels": 8,
            "with_conditioning": False,
            "transformer_num_layers": 1,
            "cross_attention_dim": None,
            "num_class_embeds": None,
            "upcast_attention": False,
            "cross_attention_dropout": 0.0,
        }

    @staticmethod
    def _validate_params(model_config: dict) -> None:
        architecture_params = model_config["architecture"]

        with_conditioning = architecture_params["with_conditioning"]
        cross_attention_dim = architecture_params["cross_attention_dim"]
        dropout_cattn = architecture_params["cross_attention_dropout"]
        num_channels = architecture_params["num_channels"]
        norm_num_groups = architecture_params["norm_num_groups"]
        attention_levels = architecture_params["attention_levels"]
        assert not (with_conditioning and cross_attention_dim is None), (
            "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
            "when using with_conditioning."
        )
        assert not (
            cross_attention_dim is not None and not with_conditioning
        ), "DiffusionModelUNet expects with_conditioning=True when specifying the cross_attention_dim."
        assert (
            0.0 <= dropout_cattn <= 1.0
        ), "Dropout cannot be negative or greater than 1.0!"

        # All number of channels should be multiples of norm_num_groups
        assert all(
            (out_channel % norm_num_groups) == 0 for out_channel in num_channels
        ), "DiffusionModelUNet expects all num_channels to be multiples of norm_num_groups."

        assert len(num_channels) == len(
            attention_levels
        ), "DiffusionModelUNet expects num_channels to be the same size as attention_levels."

    def _set_default_architecture_params(self, model_config: dict) -> dict:
        for key, value in self.architecture_default_params.items():
            if key not in model_config["architecture"]:
                model_config["architecture"][key] = value
                warn(
                    f"Parameter {key} not found in the `architecture` field of `model_config`. Setting value to default: {value}.",
                    UserWarning,
                )
        # special case for the DDPM model
        model_config["architecture"]["out_channels"] = model_config["architecture"].get(
            "out_channels", model_config["n_channels"]
        )
        warn(
            "Parameter `out_channels` not found in the `model_config`. Setting value to `n_channels`.",
            UserWarning,
        )

        return model_config
