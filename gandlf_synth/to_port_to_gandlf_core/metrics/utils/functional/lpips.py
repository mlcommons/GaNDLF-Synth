import os
import torch
from torch import nn
from torch import Tensor
import torchmetrics.functional.image as tm_image
from torchmetrics.functional.image.lpips import (
    Alexnet,
    Vgg16,
    SqueezeNet,
    NetLinLayer,
    ScalingLayer,
)
from typing import Tuple, Union, Literal, Optional, List
from acsconv.converters import ACSConverter, Conv3dConverter, SoftACSConverter


def _spatial_average(in_tens: Tensor, n_dims: int, keep_dim: bool = True) -> Tensor:
    """
    Spatial averaging over height and width of images.

    Args:
        in_tens (torch.Tensor): input tensor
        n_dims (int): number of dimensions
        keep_dim (bool, optional): whether to keep the dimensions. Defaults to True.

    Returns:
        torch.Tensor: spatially averaged tensor
    """

    if n_dims == 3:
        return in_tens.mean([2, 3, 4], keepdim=keep_dim)
    return in_tens.mean([2, 3], keepdim=keep_dim)


def _upsample(in_tens: Tensor, out_hw: Tuple[int, ...] = (64, 64)) -> Tensor:
    """
    Upsample input with bilinear interpolation.

    Args:
        in_tens (torch.Tensor): input tensor
        out_hw (Tuple[int, ...], optional): output size. Defaults to (64, 64).

    Returns:
        torch.Tensor: upsampled tensor
    """
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(in_tens)


def _normalize_tensor(in_feat: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize input tensor.

    Args:
        in_feat (torch.Tensor): input tensor
        eps (float, optional): epsilon. Defaults to 1e-8.

    Returns:
        torch.Tensor: normalized tensor
    """
    norm_factor = torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / norm_factor


def _resize_tensor(x: Tensor, size: int = 64) -> Tensor:
    """
    Resize input tensor.

    Args:
        x (torch.Tensor): input tensor
        size (int, optional): size of the output. Defaults to 64.

    Returns:
        torch.Tensor: resized tensor
    """
    if x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode="area")
    return torch.nn.functional.interpolate(
        x, (size, size), mode="bilinear", align_corners=False
    )


class _LPIPSGandlf(nn.Module):
    def __init__(
        self,
        n_dim: int = 2,
        pretrained: bool = True,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        spatial: bool = False,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = True,
        model_path: Optional[str] = None,
        eval_mode: bool = True,
        resize: Optional[int] = None,
        **kwargs,
    ):
        """
        An implementation of the LPIPS metric from torchmetrics to handle both 2D and 3D images
        in GANDLF.

        Args:
            n_dim (int, optional): number of dimensions. Defaults to 2.
            pretrained (bool, optional): whether to use pretrained weights. Defaults to True.
            net (Literal["alex", "vgg", "squeeze"], optional): network type. Defaults to "alex".
            spatial (bool, optional): whether to use spatial. Defaults to False.
            pnet_rand (bool, optional): whether to use random weights. Defaults to False.
            pnet_tune (bool, optional): whether to tune the network. Defaults to False.
            use_dropout (bool, optional): whether to use dropout. Defaults to True.
            model_path (Optional[str], optional): path to the model. Defaults to None.
            eval_mode (bool, optional): whether to use evaluation mode. Defaults to True.
            resize (Optional[int], optional): resize the input. Defaults to None.
        """
        super().__init__()
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.resize = resize

        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = Alexnet  # type: ignore[assignment]
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = SqueezeNet  # type: ignore[assignment]
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)  # type: ignore[assignment]
        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(
                    os.path.join(os.path.abspath(tm_image.__file__), "..", f"lpips_models/{net}.pth")  # type: ignore[misc]
                )

            self.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=False
            )

        if eval_mode:
            self.eval()

        self.n_dim = n_dim

    def forward(
        self,
        in0: Tensor,
        in1: Tensor,
        retperlayer: bool = False,
        normalize: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Forward pass of the network.

        Args:
            in0 (torch.Tensor): input tensor
            in1 (torch.Tensor): input tensor
            retperlayer (bool, optional): whether to return per layer. Defaults to False.
            normalize (bool, optional): whether to normalize the input. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: loss value or loss value and per layer loss values
        """

        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # normalize input
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)
        with torch.no_grad():
            outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
            feats0, feats1, diffs = {}, {}, {}

            for kk in range(self.L):
                feats0[kk], feats1[kk] = _normalize_tensor(
                    outs0[kk]
                ), _normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

            res = []
            for kk in range(self.L):
                if self.spatial:
                    res.append(
                        _upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:]))
                    )
                else:
                    res.append(
                        _spatial_average(
                            self.lins[kk](diffs[kk]), n_dims=self.n_dim, keep_dim=True
                        )
                    )
        val: Tensor = sum(res)  # type: ignore[assignment]
        if retperlayer:
            return (val, res)
        return val


class _NoTrainLpipsLPIPSGandlf(_LPIPSGandlf):
    """
    A wrapper that implements the LPIPS metric from torchmetrics to
    handle both 2D and 3D images, single and multi-channel images.
    """

    def train(self, mode: bool) -> "_NoTrainLpipsLPIPSGandlf":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor, normalize: bool) -> bool:
    """
    Check if input is valid.

    Args:
        img (torch.Tensor): input tensor
        normalize (bool): whether to normalize the input

    Returns:
        bool: whether the input is valid
    """
    value_check = (
        img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    )
    return (img.ndim == 4 or img.ndim == 5) and value_check


def lpips_update(
    img1: Tensor, img2: Tensor, net: nn.Module, normalize: bool
) -> Tuple[Tensor, Union[int, Tensor]]:
    """
    Update internal states with lpips score.

    Args:
        img1 (torch.Tensor): input tensor
        img2 (torch.Tensor): input tensor
        net (torch.nn.Module): network
        normalize (bool): whether to normalize the input

    Returns:
        Tuple[torch.Tensor, Union[int, torch.Tensor]]: loss value and total number of images

    """
    assert _valid_img(img1, normalize) and _valid_img(img2, normalize), (
        "Expected both inputs to be 4D or 5D with shape "
        "[N x C x H x W] or [N x C x D x H x W]"
        f"Got input with shape {img1.shape} and {img2.shape}."
        f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} "
        f" when all values are expected to be in the "
        f"{[0,1] if normalize else [-1,1]} range."
    )

    loss = net(img1, img2, normalize=normalize).squeeze()
    return loss, img1.shape[0]


def lpips_compute(
    sum_scores: Tensor,
    total: Union[Tensor, int],
    reduction: Literal["sum", "mean"] = "mean",
) -> Tensor:
    """
    Compute the final LPIPS score.

    Args:
        sum_scores (torch.Tensor): sum of the scores
        total (Union[torch.Tensor, int]): total number of images
        reduction (Literal["sum", "mean"], optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: final LPIPS score
    """
    return sum_scores / total if reduction == "mean" else sum_scores


def determine_converter(
    converter_type: Literal["soft", "acs", "conv3d"] = "soft"
) -> Union[SoftACSConverter, ACSConverter, Conv3dConverter]:
    """
    Determine the converter type to use for 2D to 3D conversion.

    Args:
        converter_type (Literal["soft", "acs", "conv3d"], optional): str indicating
    the type of converter to use for converting the net into a 3D network if the
    input is 5D. Choose between `'soft'`, `'asc'` or `'conv3d'`. If ``None`` will use
    `'soft'` by default.

    Returns:
        Union[None, SoftACSConverter, ACSConverter, Conv3dConverter]: the converter to use.
    """
    assert converter_type in ["soft", "acs", "conv3d", None], (
        f"Expected converter_type to be one of 'soft', 'acs', 'conv3d' or None."
        f" Got {converter_type}."
    )
    if converter_type is None or converter_type == "soft":
        converter = SoftACSConverter
    elif converter_type == "acs":
        converter = ACSConverter
    elif converter_type == "conv3d":
        converter = Conv3dConverter
    return converter


def modify_net_input(
    net: nn.Module, net_type: Literal["alex", "vgg", "squeeze"], n_channels: int
) -> None:
    """
    Modify the input layer of the network to accept the correct number
    of channels.

    Args:
        net: network
        net_type: str indicating backbone network type to use. Choose
    `'squeeze'`, not implemented for `alex` or `vgg`.
        n_channels: number of channels
    """
    if net_type == "squeeze":
        net.net.slices._modules["0"]._modules["0"] = torch.nn.Conv2d(
            n_channels, 64, kernel_size=(3, 3), stride=(2, 2)
        )
    elif net_type == "alex":
        net.net._modules["slice1"]._modules["0"] = torch.nn.Conv2d(
            n_channels, 64, kernel_size=(11, 11), stride=(4, 4)
        )

    elif net_type == "vgg":
        net.net._modules["slice1"]._modules["0"] = torch.nn.Conv2d(
            n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )


def modify_scaling_layer(net: nn.Module) -> None:
    """
    Modify the scaling layer of the network to Identity for
    cases other than 2D 3-channel images.

    Args:
        net (torch.nn.Module): network
    """
    if isinstance(net.scaling_layer, ScalingLayer):
        net.scaling_layer = nn.Identity()


def learned_perceptual_image_patch_similarity(
    img1: Tensor,
    img2: Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    reduction: Literal["sum", "mean"] = "mean",
    normalize: bool = False,
    n_dim: int = 2,
    n_channels: int = 1,
    converter_type: Literal["soft", "acs", "conv3d"] = "soft",
) -> Tensor:
    """
    Functional Interface for The Learned Perceptual Image Patch Similarity
    (`LPIPS_`) calculates perceptual similarity between two images.
    LPIPS essentially computes the similarity between the activations of
    two image patches for some pre-defined network. This measure has been
    shown to match human perception well. A low LPIPS score means that
    image patches are perceptual similar.
    Both input image patches are expected to have shape ``(N, C, H, W)``
    or ``(N, C, D, H, W)``. The minimum size of `H, W` depends on the chosen
    backbone (see `net_type` arg).

    Args:
        img1 (torch.Tensor): first set of images
        img2 (torch.Tensor): second set of images
        net_type (str, optional): backbone network type to use. Choose
    between `'alex'`, `'vgg'`, `'squeeze'`. Defaults to `'alex'`.
        reduction (str, optional): Specifies the reduction to apply to the
    output. Choose between `'sum'`, `'mean'`. Defaults to `'mean'`.
        normalize (bool, optional): by default this is ``False`` meaning that the input is
    expected to be in the [-1,1] range. If set to ``True`` will instead
    expect input to be in the ``[0,1]`` range.
        converter_type (str, optional): string indicating the type of converter to use for
    converting the net into a 3D network if the input is 5D. Choose between `'soft'`, `'acs'`, `'conv3d'`.
    Will use `'soft'` by default.

    Returns:
        torch.Tensor: The LPIPS score

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
        tensor(0.1008, grad_fn=<DivBackward0>)
    """
    net = _NoTrainLpipsLPIPSGandlf(n_dim=n_dim, net=net_type).to(
        device=img1.device, dtype=img1.dtype
    )
    if n_channels != 3:
        modify_scaling_layer(net)
        modify_net_input(net, net_type, n_channels)
    if n_dim == 3:
        modify_scaling_layer(net)
        converter = determine_converter(converter_type)
        net = converter(net)
    loss, total = lpips_update(img1, img2, net, normalize)
    return lpips_compute(loss.sum(), total, reduction)
