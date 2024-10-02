"""This implementation is an adaptation of the one found in 
https://github.com/Project-MONAI/GenerativeModels by MONAI Consortium.
It contains wrappers and class extensions for compatibility with the
gandlf-synth library.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.networks.nets.diffusion_model_unet import (
    BasicTransformerBlock,
    AttentionBlock,
)

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


from typing import Type, Optional, Tuple, Any, Iterable


class SpatialTransformerGandlf(nn.Module):
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
        cross_attention_dim (int, optional): Number of context dimensions to use. Defaults to None.
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

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]

        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(
                batch, height * width * depth, inner_dim
            )

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = (
                x.reshape(batch, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        if self.spatial_dims == 3:
            x = (
                x.reshape(batch, height, width, depth, inner_dim)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

        x = self.proj_out(x)
        return x + residual


class DownsampleGandlf(nn.Module):
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

    def forward(
        self, x: torch.Tensor, emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        return self.op(x)


class UpsampleGandlf(nn.Module):
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

    def forward(
        self, x: torch.Tensor, emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError("Input channels should be equal to num_channels")

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlockGandlf(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): number of input channels.
        temb_channels (int): number of timestep embedding channels.
        conv (Type[nn.Module]): convolution module to use.
        pool (Type[nn.Module]): pooling module to use.
        out_channels (int, optional): number of output channels. Defaults to None.
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
        out_channels: Optional[int] = None,
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

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h


class DownBlockGandlf(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlockGandlf(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlockGandlf(nn.Module):
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
        cross_attention_dim (int, optional): number of context dimensions to use. Defaults to None.
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
        cross_attention_dim: Optional[int] = None,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlockGandlf(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class CrossAttnMidBlockGandlf(nn.Module):
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
        cross_attention_dim (int, optional): number of context dimensions to use. Defaults to None.
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
        cross_attention_dim: Optional[int] = None,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlockGandlf(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AttnUpBlockGandlf(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlockGandlf(nn.Module):
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
        cross_attention_dim (int ): number of context dimensions to use.
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
        cross_attention_dim: Optional[int] = None,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


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
    cross_attention_dim: Optional[int] = None,
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
            pool=pool,
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
    cross_attention_dim: Optional[int] = None,
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
    cross_attention_dim: Optional[int] = None,
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


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


def convert_to_tuple(value: Any, size: int) -> Tuple[int, ...]:
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,) * size


class DDPM(ModelBase):
    def __init__(self, model_config: Type[AbstractModelConfig]) -> None:
        ModelBase.__init__(self, model_config)

        num_channels = model_config.architecture["num_channels"]
        out_channels = model_config.architecture["out_channels"]
        num_res_blocks = model_config.architecture["num_res_blocks"]
        norm_num_groups = model_config.architecture["norm_num_groups"]
        norm_eps = model_config.architecture["norm_eps"]
        resblock_updown = model_config.architecture["resblock_updown"]
        attention_levels = model_config.architecture["attention_levels"]
        num_head_channels = model_config.architecture["num_head_channels"]
        transformer_num_layers = model_config.architecture["transformer_num_layers"]
        cross_attention_dim = model_config.architecture["cross_attention_dim"]
        upcast_attention = model_config.architecture["upcast_attention"]
        dropout_cattn = model_config.architecture["cross_attention_dropout"]

        self.num_class_embeds = model_config.architecture["num_class_embeds"]
        self.with_conditioning = model_config.architecture["with_conditioning"]
        self.block_out_channels = num_channels

        if isinstance(num_head_channels, int):
            num_head_channels = convert_to_tuple(
                num_head_channels, len(attention_levels)
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = convert_to_tuple(num_res_blocks, len(num_channels))
        self.conv_in = self.Conv(
            in_channels=self.n_channels,
            out_channels=num_channels[0],
            stride=1,
            kernel_size=3,
            padding=1,
        )
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_class_embeds is not None:
            self.class_embedding = nn.Embedding(self.num_class_embeds, time_embed_dim)
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]

        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=self.n_dimensions,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not self.with_conditioning),
                with_cross_attn=(attention_levels[i] and self.with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=False,  # no flash attention support for now
                dropout_cattn=dropout_cattn,
                conv=self.Conv,
                pool=self.AvgPool,
            )
            self.down_blocks.append(down_block)

        self.middle_block = get_mid_block(
            spatial_dims=self.n_dimensions,
            in_channels=num_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=self.with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=False,
            dropout_cattn=dropout_cattn,
            conv=self.Conv,
            pool=self.AvgPool,
        )
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(num_channels) - 1)
            ]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=self.n_dimensions,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not self.with_conditioning),
                with_cross_attn=(
                    reversed_attention_levels[i] and self.with_conditioning
                ),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=False,
                dropout_cattn=dropout_cattn,
                conv=self.Conv,
                pool=self.AvgPool,
            )

            self.up_blocks.append(up_block)

        self.out = nn.Sequential(
            nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=num_channels[0],
                eps=norm_eps,
                affine=True,
            ),
            nn.SiLU(),
            self.Conv(
                in_channels=num_channels[0],
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks (N, C, FeatureMapsDims).
            mid_block_additional_residual: additional residual tensor for mid block (N, C, FeatureMapsDims).
        """
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        h = self.conv_in(x)

        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(
                hidden_states=h, temb=emb, context=context
            )
            for residual in res_samples:
                down_block_res_samples.append(residual)

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            h = upsample_block(
                hidden_states=h,
                res_hidden_states_list=res_samples,
                temb=emb,
                context=context,
            )

        h = self.out(h)

        return h
