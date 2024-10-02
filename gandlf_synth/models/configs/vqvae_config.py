from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class VQVAEConfig(AbstractModelConfig):
    """
    Configuartion class for the VQVAE model, used for reconstruction tasks.

    """

    @staticmethod
    def _prepare_default_model_params() -> dict:
        return {}

    @staticmethod
    def _prepare_default_architecture_params() -> dict:
        return {
            "embedding_dim": 64,
            "num_channels_upsample_downsample_layers": (96, 96, 192),
            "num_residual_layers": 3,
            "num_residual_channels": (96, 96, 192),
            "downsample_conv_parameters": ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            "upsample_conv_parameters": (
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
            ),
            "dropout": 0.0,
            "num_embeddings": 32,
            "loss_scaling_commitment_cost": 0.25,
            "EMA_decay": 0.5,
            "epsilon": 1e-5,
        }

    @staticmethod
    def _validate_params(model_config: dict) -> None:
        num_residual_channels = model_config["architecture"]["num_residual_channels"]
        num_residual_layers = model_config["architecture"]["num_residual_layers"]
        downsample_conv_parameters = model_config["architecture"][
            "downsample_conv_parameters"
        ]
        upsample_conv_parameters = model_config["architecture"][
            "upsample_conv_parameters"
        ]
        num_channels_upsample_downsample_layers = model_config["architecture"][
            "num_channels_upsample_downsample_layers"
        ]

        assert (
            len(num_residual_channels) == num_residual_layers
        ), "Number of elements in `num_residual_channels` should be equal to `num_residual_layers`."
        for parameter in downsample_conv_parameters:
            assert (
                len(parameter) == 4
            ), "`downsample_conv_parameters` in model `architecture` config should be a tuple of tuples with 4 integers."
        for parameter in upsample_conv_parameters:
            assert (
                len(parameter) == 5
            ), "`upsample_conv_parameters` in model `architecture` config should be a tuple of tuples with 5 integers."

        assert len(downsample_conv_parameters) == len(
            num_channels_upsample_downsample_layers
        ), "Number of downsample layers should be equal to number of upsample layers."
        assert len(upsample_conv_parameters) == len(
            num_channels_upsample_downsample_layers
        ), "Number of upsample layers should be equal to number of downsample layers."
