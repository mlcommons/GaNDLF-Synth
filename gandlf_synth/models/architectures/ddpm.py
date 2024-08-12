"""This implementation is an adaptation of the one found in 
https://github.com/Project-MONAI/GenerativeModels by MONAI Consortium.
It contains wrappers and class extensions for compatibility with the
gandlf-synth library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from generative.networks.layers.vector_quantizer import VectorQuantizer, EMAQuantizer
from generative.networks.nets.diffusion_model_unet import (
    CrossAttention,
    BasicTransformerBlock,
    AttentionBlock,
    SpatialTransformer,
    Downsample,
    Upsample,
    ResnetBlock,
    DownBlock,
    AttnDownBlock,
    CrossAttnDownBlock,
    AttnMidBlock,
    CrossAttnMidBlock,
    UpBlock,
    AttnUpBlock,
    CrossAttnUpBlock,
)

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


from typing import Type, Union, List, Tuple, Optional, Dict


# to rewrite (containing convolutions)
# AttnUpBlock, CrossAttnUpBlock, get_down_block,get_mid_block
# get_up_block, DiffusionModelUNet


class SpatialTransformerGandlf(SpatialTransformer):

    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        spatial_dims (int): Number of spatial dimensions.
        in_channels (int): Number of channels in the input and output.
        num_attention_heads (int): Number of heads to use for multi-head attention.
        num_head_channels (int): Number of channels in each attention head.
        conv (Type[nn.Module]): Convolution module to use.
        num_layers (int, optional): Number of layers of Transformer blocks to use. Defaults to 1.
        dropout (float, optional): Dropout probability to use. Defaults to 0.0.
        norm_num_groups (int, optional): Number of groups for the normalization. Defaults to 32.
        norm_eps (float, optional): Epsilon for the normalization. Defaults to 1e-6.
        cross_attention_dim (int | None, optional): Number of context dimensions to use. Defaults to None.
        upcast_attention (bool, optional): If True, upcast attention operations to full precision. Defaults to False.
        use_flash_attention (bool, optional): If True, use flash attention for a memory efficient attention mechanism. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        conv: Type[nn.Module],
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: Optional[int] = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )

        self.proj_in = conv(
            in_channels=in_channels,
            out_channels=inner_dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = conv(
            in_channels=inner_dim,
            out_channels=in_channels,
            stride=1,
            kernel_size=1,
            padding=0,
        )


class DownsampleGandlf(Downsample):
    """
    Downsampling layer.

    Args:
        num_channels (int): Number of input channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        use_conv (bool): If True, uses Convolution instead of Pool average to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels (int, optional): Number of output channels. Defaults to None.
        padding (int, optional): Controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to 1.
    """

    def __init__(
        self,
        num_channels: int,
        use_conv: bool,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = conv(
                in_channels=num_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
            )

        else:
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must be equal when use_conv=False"
                )
            self.op = pool(kernel_size=2, stride=2)


class UpsampleGandlf(Upsample):
    """
    Upsampling layer with an optional convolution.

    Args:
        num_channels (int): Number of input channels.
        use_conv (bool): If True, uses Convolution instead of Pool average to perform downsampling.
        conv (Type[nn.Module]): Convolution module to use.
        out_channels (int, optional): Number of output channels. Defaults to None.
        padding (int, optional): Controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension. Defaults to 1.
    """

    def __init__(
        self,
        num_channels: int,
        use_conv: bool,
        conv: Type[nn.Module],
        out_channels: Optional[int] = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        self.conv = None
        if use_conv:
            self.conv = conv(
                in_channels=num_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
            )


class ResnetBlockGandlf(ResnetBlock):

    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): convolution module to use.
        pool (Type[nn.Module]): pooling module to use.
        out_channels (int | None, optional): number of output channels. Defaults to None.
        up (bool, optional): if True, performs upsampling. Defaults to False.
        down (bool, optional): if True, performs downsampling. Defaults to False.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        out_channels: int | None = None,
        up: Optional[bool] = False,
        down: Optional[bool] = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True,
        )
        self.nonlinearity = nn.SiLU()
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=self.out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleGandlf(in_channels, use_conv=False)
        elif down:
            self.downsample = DownsampleGandlf(
                in_channels, conv=conv, pool=pool, use_conv=False
            )

        self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.out_channels,
            eps=norm_eps,
            affine=True,
        )
        self.conv2 = conv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv(
                in_channels=in_channels,
                out_channels=self.out_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )


class DownBlockGandlf(DownBlock):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 1.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        add_downsample (bool, optional): if True add downsample block. Defaults to True.
        resblock_updown (bool, optional): if True use residual blocks for downsampling. Defaults to False.
        downsample_padding (int, optional): padding used in the downsampling block. Defaults to 1.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    conv=conv,
                    pool=pool,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DownsampleGandlf(
                    num_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None


class AttnDownBlockGandlf(AttnDownBlock):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding  channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 1.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        add_downsample (bool, optional): if True add downsample block. Defaults to True.
        resblock_updown (bool, optional): if True use residual blocks for downsampling. Defaults to False.
        downsample_padding (int, optional): padding used in the downsampling block. Defaults to 1.
        num_head_channels (int, optional): number of channels in each attention head. Defaults to 1.
        use_flash_attention (bool, optional): if True, use flash attention for a memory efficient attention mechanism. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DownsampleGandlf(
                    num_channels=out_channels,
                    use_conv=True,
                    conv=conv,
                    pool=pool,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None


class CrossAttnDownBlockGandlf(CrossAttnDownBlock):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.

    Args:
        spatial_dims (int): number of spatial dimensions.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 1.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        add_downsample (bool, optional): if True add downsample block. Defaults to True.
        resblock_updown (bool, optional): if True use residual blocks for downsampling. Defaults to False.
        downsample_padding (int, optional): padding used in the downsampling block. Defaults to 1.
        num_head_channels (int, optional): number of channels in each attention head. Defaults to 1.
        transformer_num_layers (int, optional): number of layers of Transformer blocks to use. Defaults to 1.
        cross_attention_dim (int | None, optional): number of context dimensions to use. Defaults to None.
        upcast_attention (bool, optional): if True, upcast attention operations to full precision. Defaults to False.
        use_flash_attention (bool, optional): if True, use flash attention for a memory efficient attention mechanism. Defaults to False.
        dropout_cattn (float, optional): if different from zero, this will be the dropout value for the cross-attention layers. Defaults to 0.0.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

            attentions.append(
                SpatialTransformerGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    conv=conv,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DownsampleGandlf(
                    num_channels=out_channels,
                    use_conv=True,
                    conv=conv,
                    pool=pool,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None


class AttnMidBlockGandlf(AttnMidBlock):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        num_head_channels (int, optional): number of channels in each attention head. Defaults to 1.
        use_flash_attention (bool, optional): if True, use flash attention for a memory efficient attention mechanism. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            conv=conv,
            pool=pool,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = ResnetBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            conv=conv,
            pool=pool,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )


class CrossAttnMidBlockGandlf(CrossAttnMidBlock):
    """
    Unet's mid block containing resnet and cross-attention blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        temb_channels (int): number of timestep embedding channels
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        num_head_channels (int, optional): number of channels in each attention head. Defaults to 1.
        transformer_num_layers (int, optional): number of layers of Transformer blocks to use. Defaults to 1.
        cross_attention_dim (int | None, optional): number of context dimensions to use. Defaults to None.
        upcast_attention (bool, optional): if True, upcast attention operations to full precision. Defaults to False.
        use_flash_attention (bool, optional): if True, use flash attention for a memory efficient attention mechanism. Defaults to False.
        dropout_cattn (float, optional): if different from zero, this will be the dropout value for the cross-attention layers. Defaults to 0.0.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = None

        self.resnet_1 = ResnetBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            conv=conv,
            pool=pool,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformerGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            conv=conv,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout=dropout_cattn,
        )
        self.resnet_2 = ResnetBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            conv=conv,
            pool=pool,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )


class UpBlockGandlf(UpBlock):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        prev_output_channel (int): number of channels from residual connection.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 1.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        add_upsample (bool, optional): if True add downsample block. Defaults to True.
        resblock_updown (bool, optional): if True use residual blocks for upsampling. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    conv=conv,
                    pool=pool,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = UpsampleGandlf(
                    num_channels=out_channels,
                    use_conv=True,
                    conv=conv,
                    out_channels=out_channels,
                )
        else:
            self.upsampler = None


class AttnUpBlockGandlf(AttnUpBlock):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        prev_output_channel (int): number of channels from residual connection.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 1.
        norm_num_groups (int, optional): number of groups for the group normalization. Defaults to 32.
        norm_eps (float, optional): epsilon for the group normalization. Defaults to 1e-6.
        add_upsample (bool, optional): if True add downsample block. Defaults to True.
        resblock_updown (bool, optional): if True use residual blocks for upsampling. Defaults to False.
        num_head_channels (int, optional): number of channels in each attention head. Defaults to 1.
        use_flash_attention (bool, optional): if True, use flash attention for a memory efficient attention mechanism. Defaults to False.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = UpsampleGandlf(
                    num_channels=out_channels,
                    conv=conv,
                    use_conv=True,
                    out_channels=out_channels,
                )
        else:
            self.upsampler = None


class CrossAttnUpBlockGandlf(CrossAttnUpBlock):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        prev_output_channel (int): number of channels from residual connection.
        out_channels (int): number of output channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): Convolution module to use.
        pool (Type[nn.Module]): Pooling module to use.
        num_res_blocks (int): number of residual blocks.
        norm_num_groups (int): number of groups for the group normalization.
        norm_eps (float): epsilon for the group normalization.
        add_upsample (bool): if True add downsample block.
        resblock_updown (bool): if True use residual blocks for upsampling.
        num_head_channels (int): number of channels in each attention head.
        transformer_num_layers (int): number of layers of Transformer blocks to use.
        cross_attention_dim (int | None): number of context dimensions to use.
        upcast_attention (bool): if True, upcast attention operations to full precision.
        use_flash_attention (bool): if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn (float): if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        conv: Type[nn.Module],
        pool: Type[nn.Module],
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = (
                in_channels if (i == num_res_blocks - 1) else out_channels
            )
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformerGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    conv=conv,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlockGandlf(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    conv=conv,
                    pool=pool,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = UpsampleGandlf(
                    num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None


def get_down_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_downsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    conv: Type[nn.Module],
    pool: Type[nn.Module],
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_attn:
        return AttnDownBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return DownBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            conv=conv,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
    spatial_dims: int,
    in_channels: int,
    temb_channels: int,
    norm_num_groups: int,
    norm_eps: float,
    with_conditioning: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    conv: Type[nn.Module],
    pool: Type[nn.Module],
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return AttnMidBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )


def get_up_block(
    spatial_dims: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    num_res_blocks: int,
    norm_num_groups: int,
    norm_eps: float,
    add_upsample: bool,
    resblock_updown: bool,
    with_attn: bool,
    with_cross_attn: bool,
    num_head_channels: int,
    transformer_num_layers: int,
    conv: Type[nn.Module],
    pool: Type[nn.Module],
    cross_attention_dim: int | None,
    upcast_attention: bool = False,
    use_flash_attention: bool = False,
    dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_attn:
        return AttnUpBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnUpBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            conv=conv,
            pool=pool,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return UpBlockGandlf(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            conv=conv,
            pool=pool,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
        )
