import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.configs.config_abc import AbstractModelConfig

from typing import Type, List, Tuple


class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
    Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
    that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
    58d9a2746493717a7c9252938da7efa6006f3739.

    This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
    to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
    on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

    Args:
        spatial_dims :  number of spatial spatial_dims.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
        decay: EMA decay. Defaults to 0.99.
        epsilon: epsilon value. Defaults to 1e-5.
        embedding_init: initialization method for the codebook. Defaults to "normal".
        ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings

        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )

        self.embedding: torch.nn.Embedding = torch.nn.Embedding(
            self.num_embeddings, self.embedding_dim
        )
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(
                self.embedding.weight.data, mode="fan_in", nonlinearity="linear"
            )
        self.embedding.weight.requires_grad = False

        self.commitment_cost: float = commitment_cost

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

        self.decay: float = decay
        self.epsilon: float = epsilon

        self.ddp_sync: bool = ddp_sync

        # Precalculating required permutation shapes
        self.flatten_permutation: Sequence[int] = (
            [0] + list(range(2, self.spatial_dims + 2)) + [1]
        )
        self.quantization_permutation: Sequence[int] = [
            0,
            self.spatial_dims + 1,
        ] + list(range(1, self.spatial_dims + 1))

    def quantize(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs: Encoding space tensors

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
            torch.Tensor: Quantization indices of shape [B,D,H,W,1]

        """

        with torch.cuda.amp.autocast(enabled=False):
            encoding_indices_view = list(inputs.shape)
            del encoding_indices_view[1]
            # inputs = inputs.float()

            # Converting to channel last format
            flat_input = (
                inputs.permute(self.flatten_permutation)
                .contiguous()
                .view(-1, self.embedding_dim)
            )

            # Calculate Euclidean distances
            distances = (
                (flat_input**2).sum(dim=1, keepdim=True)
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(flat_input, self.embedding.weight.t())
            )

            # Mapping distances to indexes
            encoding_indices = torch.max(-distances, dim=1)[1]
            encodings = torch.nn.functional.one_hot(
                encoding_indices, self.num_embeddings
            ).type_as(inputs)

            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices: Tensor in channel last format which holds indices referencing atomic
                elements from self.embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        with torch.cuda.amp.autocast(enabled=False):
            return (
                self.embedding(embedding_indices)
                .permute(self.quantization_permutation)
                .contiguous()
            )

    @torch.jit.unused
    def distributed_synchronization(
        self, encodings_sum: torch.Tensor, dw: torch.Tensor
    ) -> None:
        """
        TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
        example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

        Args:
            encodings_sum: The summation of one hot representation of what encoding was used for each
                position.
            dw: The multiplication of the one hot representation of what encoding was used for each
                position with the flattened input.

        """
        if self.ddp_sync and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)
        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(
                    torch.mul(encodings_sum, 1 - self.decay)
                )

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))

        # Encoding Loss
        loss = self.commitment_cost * torch.nn.functional.mse_loss(
            quantized.detach(), inputs
        )

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class VectorQuantizer(torch.nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation. Defaults to None
    """

    def __init__(self, quantizer: torch.nn.Module = None):
        super().__init__()

        self.quantizer: torch.nn.Module = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices = self.quantizer(inputs)

        # Perplexity calculations
        avg_probs = (
            torch.histc(
                encoding_indices.float(),
                bins=self.quantizer.num_embeddings,
                max=self.quantizer.num_embeddings,
            )
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return loss, quantized

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices = self.quantizer(encodings)

        return encoding_indices


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
            dropout(dropout_prob),
            nn.ReLU(),
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
                    in_channels=(
                        in_channels if i == 0 else num_channels_downsample_layers[i - 1]
                    ),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels_upsample_layers: Sequence[int],
        num_residual_layers: int,
        num_residual_channels: int,
        upsample_conv_parameters: List[Tuple[int, int, int, int, int]],
        dropout_prob: float,
        dropout: nn.Module,
        conv: nn.Module,
        conv_transpose: nn.Module,
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
            conv_transpose (nn.Module): The transposed convolution layer.
        """
        super().__init__()
        blocks = []
        blocks.append(
            conv(
                in_channels=in_channels,
                out_channels=num_channels_upsample_layers[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

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
            blocks.append(
                conv_transpose(
                    in_channels=num_channels_upsample_layers[i],
                    out_channels=(
                        out_channels
                        if i == len(num_channels_upsample_layers) - 1
                        else num_channels_upsample_layers[i + 1]
                    ),
                    stride=upsample_conv_parameters[i][0],
                    kernel_size=upsample_conv_parameters[i][1],
                    dilation=upsample_conv_parameters[i][2],
                    padding=upsample_conv_parameters[i][3],
                    output_padding=upsample_conv_parameters[i][4],
                )
            )
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
            num_channels_downsample_layers=model_config.architecture[
                "num_channels_upsample_downsample_layers"
            ],
            num_residual_layers=model_config.architecture["num_residual_layers"],
            num_residual_channels=model_config.architecture["num_residual_channels"],
            downsample_conv_parameters=model_config.architecture[
                "downsample_conv_parameters"
            ],
            dropout_prob=model_config.architecture["dropout"],
            dropout=self.Dropout,
            conv=self.Conv,
        )
        reversed_num_channels_upsample_downsample_layers = list(
            reversed(
                model_config.architecture["num_channels_upsample_downsample_layers"]
            )
        )
        reversed_num_residual_channels = list(
            reversed(model_config.architecture["num_residual_channels"])
        )

        self.decoder = _DecoderVQVAE(
            in_channels=model_config.architecture["embedding_dim"],
            out_channels=self.n_channels,
            num_channels_upsample_layers=reversed_num_channels_upsample_downsample_layers,
            num_residual_layers=model_config.architecture["num_residual_layers"],
            num_residual_channels=reversed_num_residual_channels,
            upsample_conv_parameters=model_config.architecture[
                "upsample_conv_parameters"
            ],
            dropout_prob=model_config.architecture["dropout"],
            dropout=self.Dropout,
            conv=self.Conv,
            conv_transpose=self.ConvTranspose,
        )
        # I on purpose ommited the embedding_init parameter here
        self.quantizer = VectorQuantizer(
            EMAQuantizer(
                spatial_dims=self.n_dimensions,
                num_embeddings=model_config.architecture["num_embeddings"],
                embedding_dim=model_config.architecture["embedding_dim"],
                commitment_cost=model_config.architecture[
                    "loss_scaling_commitment_cost"
                ],
                decay=model_config.architecture["EMA_decay"],
                epsilon=model_config.architecture["epsilon"],
                ddp_sync=False,  # FIXME! Attention needed, if this is properly synchronizing with all supported dist strategies
            )
        )
        self.full_module = nn.Sequential(self.encoder, self.quantizer, self.decoder)

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
        # part of the loss is returned here, the rest is calculated in the training loop
        return x_recon, quantization_loss
