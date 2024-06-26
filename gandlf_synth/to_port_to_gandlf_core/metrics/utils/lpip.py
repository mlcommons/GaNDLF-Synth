from typing import Any, ClassVar, List
from typing_extensions import Literal
import torch
from torch import Tensor
from torchmetrics.metric import Metric

from .functional import (
    lpips_compute,
    lpips_update,
    determine_converter,
    modify_net_input,
    modify_scaling_layer,
    _NoTrainLpipsLPIPSGandlf,
)


class LPIPSGandlf(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_scores: Tensor
    total: Tensor
    feature_network: str = "net"

    # due to the use of named tuple in the backbone the net variable cannot be scripted
    __jit_ignored_attributes__: ClassVar[List[str]] = ["net"]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        n_dim: int = 2,
        n_channels: int = 1,
        converter_type: Literal["soft", "acs", "conv3d"] = "soft",
        **kwargs: Any,
    ):
        """
        Initialize the LPIPS metric for GanDLF. This metric is based on the
        torchmetrics implementation of LPIPS, with modifications to allow usage
        of single channel data and 3D data. Note that it uses the pre-trained
        model from the torchmetrics implementation, originally designed
        for 3-channel 2D data. Here the layers are modified, so results need
        to be interpreted with caution. For 2D 3-channel data, the results
        are expected to be similar to the original implementation.

        Args:
            net_type (Literal["vgg", "alex", "squeeze"]): The network type. Defaults to 'alex'.
            reduction (Literal["sum", "mean"]): The reduction type, one of 'mean' or 'sum'.
        Defaults to 'mean'.
            normalize (bool): Whether to normalize the input images. Defaults to False.
            n_dim (int): The number of dimensions of the input images. Defaults to 2.
            n_channels (int): The number of channels of the input images. Defaults to 1.
            converter_type (Literal["soft","acs", "conv3d]: The converter type
        from ACS, one of 'soft','acs' or 'conv3d'. Defaults to 'soft'.
            **kwargs: Additional arguments for the metric.
        """

        super().__init__(**kwargs)

        valid_net_type = ("vgg", "alex", "squeeze")
        assert (
            net_type in valid_net_type
        ), f"Invalid net_type: {net_type}, expected one of {valid_net_type}"
        self.net = _NoTrainLpipsLPIPSGandlf(n_dim=n_dim, net=net_type)

        valid_reduction = ("mean", "sum")
        assert (
            reduction in valid_reduction
        ), f"Invalid reduction: {reduction}, expected one of {valid_reduction}"

        self.reduction = reduction
        assert isinstance(
            normalize, bool
        ), f"normalize should be a bool, got {normalize}"
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")
        if n_channels != 3:
            modify_scaling_layer(self.net)
            modify_net_input(self.net, net_type, n_channels)
        if n_dim == 3:
            modify_scaling_layer(self.net)
            converter = determine_converter(converter_type)
            converter(self.net)

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """
        Update internal states with lpips score.

        Args:
            img1 (torch.Tensor): The first image tensor.
            img2 (torch.Tensor): The second image tensor.
        """
        loss, total = lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        self.sum_scores += loss.sum()
        self.total += total

    def compute(self) -> Tensor:
        """
        Compute final perceptual similarity metric.

        Returns:
            torch.Tensor: The LPIPS score.
        """
        return lpips_compute(self.sum_scores, self.total, self.reduction)


if __name__ == "__main__":
    calc = LPIPSGandlf()
