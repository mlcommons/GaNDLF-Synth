import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from generative.networks.layers.vector_quantizer import VectorQuantizer, EMAQuantizer
from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.configs.config_abc import AbstractModelConfig

from typing import Type, Union, List, Tuple

class _ResidualBlockVQVAE(nn.Module):
    """
    Residual block for VQVAE model.
    """

    def __init__(
        self,
        in_channels: int,
        num_residual_channels: int,
        dropout_prob: float,
        dropout: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initialize the residual block.

        Args:
            in_channels (int): The number of input channels.
            num_residual_channels (int): The number of residual channels.
            dropout_prob (float): The dropout probability.
            dropout (nn.Module): The dropout layer.
            conv (nn.Module): The convolution layer.
        """

        super().__init__()

        self.residual_block = nn.Sequential(
            conv(
                in_channels, num_residual_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            dropout(dropout_prob),
            conv(
                num_residual_channels, in_channels, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block
        """
        return F.relu(x + self.residual_block(x), inplace=True)


class _EncoderVQVAE(nn.Module):
    """
    Encoder for VQVAE model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels_downsample_layers: Sequence[int],
        num_residual_layers: int,
        num_residual_channels: int,
        downsample_conv_parameters: List[Tuple[int, int, int, int]],
        dropout_prob: float,
        dropout: nn.Module,
        conv: nn.Module,
    ):
        """
        Initialize the encoder for VQVAE model.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            num_channels_downsample_layers (Sequence[int]): The number of channels in the consecutive downsample layers.
            num_residual_layers (int): The number of residual layers.
            num_residual_channels (int): The number of residual channels.
            downsample_conv_parameters (List[Tuple[int, int, int, int]]): The parameters for the downsample convolution layers. Each tuple
        specifies the param for given layer, with the values corresponding to: stride, kernel_size, dilation, padding.
            dropout_prob (float): The dropout probability.
            dropout (nn.Module): The dropout layer.
            conv (nn.Module): The convolution layer.
        """
        super().__init__()
        blocks = []
        for i in range(len(num_channels_downsample_layers)):
            blocks.append(
                conv(
                    in_channels=in_channels
                    if i == 0
                    else num_channels_downsample_layers[i - 1],
                    out_channels=num_channels_downsample_layers[i],
                    stride=downsample_conv_parameters[i][0],
                    kernel_size=downsample_conv_parameters[i][1],
                    dilation=downsample_conv_parameters[i][2],
                    padding=downsample_conv_parameters[i][3],
                )
            )
            if i > 0:
                blocks.append(dropout(dropout_prob))
            blocks.append(nn.ReLU())
            for _ in range(num_residual_layers):
                blocks.append(
                    _ResidualBlockVQVAE(
                        in_channels=num_channels_downsample_layers[i],
                        num_residual_channels=num_residual_channels[i],
                        dropout_prob=dropout_prob,
                        dropout=dropout,
                        conv=conv,
                    )
                )
        blocks.append(
            conv(
                in_channels=num_channels_downsample_layers[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.encoder_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder
        """
        for block in self.encoder_blocks:
            x = block(x)
        return x


class _DecoderVQVAE(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        num_channels_upsample_layers: Sequence[int],
        num_residual_layers: int,
        num_residual_channels: int,
        upsample_conv_parameters: List[Tuple[int, int, int, int, int]],
        dropout_prob: float,
        dropout: nn.Module,
        conv: nn.Module,
    ):
        """
        Decoder for VQVAE model.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            num_channels_upsample_layers (Sequence[int]): The number of channels in the consecutive upsample layers.
        It is assumed that this is the reverse of the argument `num_channels_downsample_layers` in the encoder.
            num_residual_layers (int): The number of residual layers.
            num_residual_channels (int): The number of residual channels. It is assumed that this is the reverse of the argument
        `num_residual_channels` in the encoder.
            upsample_conv_parameters (List[Tuple[int, int, int, int, int]]): The parameters for the upsample convolution layers. Each tuple
        specifies the param for given layer, with the values corresponding to: stride, kernel_size, dilation, padding, output_padding.
            dropout_prob (float): The dropout probability.
            dropout (nn.Module): The dropout layer.
            conv (nn.Module): The convolution layer.
        """
        super().__init__()
        blocks = []
        blocks.append(conv(
            in_channels=in_channels,
            out_channels=num_channels_upsample_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
        ))

        for i in range(len(num_channels_upsample_layers)):
            for _ in range(num_residual_layers):
                blocks.append(
                    _ResidualBlockVQVAE(
                        in_channels=num_channels_upsample_layers[i],
                        num_residual_channels=num_residual_channels[i],
                        dropout_prob=dropout_prob,
                        dropout=dropout,
                        conv=conv,
                    )
                )
                blocks.append(dropout(dropout_prob))
            blocks.append(conv(
                in_channels=num_channels_upsample_layers[i],
                out_channels=out_channels if i == len(num_channels_upsample_layers) - 1 else num_channels_upsample_layers[i + 1],
                stride=upsample_conv_parameters[i][0],
                kernel_size=upsample_conv_parameters[i][1],
                dilation=upsample_conv_parameters[i][2],
                padding=upsample_conv_parameters[i][3],
                output_padding=upsample_conv_parameters[i][4],
            ))
            if i < len(num_channels_upsample_layers) - 1:
                blocks.append(dropout(dropout_prob))
                blocks.append(nn.ReLU())
        self.decoder_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder
        """
        for block in self.decoder_blocks:
            x = block(x)
        return x


class VQVAE(ModelBase):

    def __init__(self, model_config: Type[AbstractModelConfig]) -> None:
        ModelBase.__init__(self, model_config)

        self.encoder = _EncoderVQVAE(
            in_channels=self.n_channels,
            out_channels=model_config.architecture["embedding_dim"],
            num_channels_downsample_layers=model_config.architecture["num_channels_upsample_downsample_layers"],
            num_residual_layers=model_config.architecture["num_residual_layers"],
            num_residual_channels=model_config.architecture["num_residual_channels"],
            downsample_conv_parameters=model_config.architecture["downsample_conv_parameters"],
            dropout_prob=model_config.architecture["dropout"],
            dropout=self.Dropout,
            conv=self.Conv,
        )
        reversed_num_channels_upsample_downsample_layers = list(reversed(model_config.architecture["num_channels_upsample_downsample_layers"]))
        reversed_num_residual_channels = list(reversed(model_config.architecture["num_residual_channels"]))

        self.decoder = _DecoderVQVAE(
            in_channels=model_config.architecture["embedding_dim"],
            out_channels=self.n_channels,
            num_channels_upsample_layers=reversed_num_channels_upsample_downsample_layers,
            num_residual_layers=model_config.architecture["num_residual_layers"],
            num_residual_channels=reversed_num_residual_channels,
            upsample_conv_parameters=model_config.architecture["upsample_conv_parameters"],
            dropout_prob=model_config.architecture["dropout"],
            dropout=self.Dropout,
            conv=self.ConvTranspose,
        )
        # I on purpose ommited the embedding_init parameter here
        self.quantizer = VectorQuantizer(
            EMAQuantizer(
                spatial_dims=self.n_dimensions,
                num_embeddings=model_config.architecture["num_embeddings"],
                embedding_dim=model_config.architecture["embedding_dim"],
                commitment_cost=model_config.architecture["loss_scaling_commitment_cost"],
                decay=model_config.architecture["loss_scaling_decay"],
                epsilon=model_config.architecture["epsilon"],
                ddp_sync=True
            )
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantization_loss, quantized = self.quantizer(x)
        return quantized, quantization_loss
    def decode(self, quantized_encodings: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized_encodings)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        quantized, quantization_loss = self.quantize(x)
        x_recon = self.decode(quantized)
        return x_recon, quantization_loss

if __name__ == "__main__":
    # Test the encoder
    spatial_dims = int
    in_channels = int
    out_channels = int
    num_channels = (96, 96, 192)
    num_res_layers = 3
    num_res_channels = (96, 96, 192)
    downsample_parameters = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1))
    upsample_parameters = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0))
    from generative.networks.nets.vqvae import Encoder, Decoder

    enc_generative = Encoder(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        num_channels=num_channels,
        num_res_layers=num_res_layers,
        num_res_channels=num_res_channels,
        downsample_parameters=downsample_parameters,
        dropout=0.1,
        act="relu",
    )
    enc_custom = _EncoderVQVAE(
        in_channels=3,
        out_channels=3,
        num_channels_downsample_layers=num_channels,
        num_residual_layers=num_res_layers,
        num_residual_channels=num_res_channels,
        downsample_conv_parameters=downsample_parameters,
        dropout_prob=0.1,
        dropout=nn.Dropout,
        conv=nn.Conv2d,
    )
    x = torch.randn(1, 3, 128, 128)
    y = enc_generative(x)
    y_custom = enc_custom(x)

    print(y.shape)
    print(y_custom.shape)

    print("DECODER")
    decoder_generative = Decoder(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        num_channels=num_channels,
        num_res_layers=num_res_layers,
        num_res_channels=num_res_channels,
        upsample_parameters=upsample_parameters,
        dropout=0.1,
        act="relu",
        output_act=None
    )
    reverse_channels = list(reversed(num_channels))
    reverse_res_channels = list(reversed(num_res_channels))
    dec_custom = _DecoderVQVAE(
        in_channels=3,
        out_channels=3,
        num_channels_upsample_layers=reverse_channels,
        num_residual_layers=num_res_layers,
        num_residual_channels=reverse_res_channels,
        upsample_conv_parameters=upsample_parameters,
        dropout_prob=0.1,
        dropout=nn.Dropout,
        conv=nn.ConvTranspose2d,
    )

    random_input = torch.randn(1, 3, 16, 16)
    y_dec = decoder_generative(random_input)
    y_custom_dec = dec_custom(random_input)
    print(y_dec.shape)
    print(y_custom_dec.shape)