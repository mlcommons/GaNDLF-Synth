from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import adaptive_avg_pool2d
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _MATPLOTLIB_AVAILABLE,
    _TORCH_FIDELITY_AVAILABLE,
)

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FrechetInceptionDistance.plot"]

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import (
        FeatureExtractorInceptionV3 as _FeatureExtractorInceptionV3,
    )
    from torch_fidelity.helpers import vassert
    from torch_fidelity.interpolate_compat_tensorflow import (
        interpolate_bilinear_2d_like_tensorflow1x,
    )
else:

    class _FeatureExtractorInceptionV3(Module):  # type: ignore[no-redef]
        pass

    vassert = None
    interpolate_bilinear_2d_like_tensorflow1x = None

    __doctest_skip__ = ["FrechetInceptionDistance", "FrechetInceptionDistance.plot"]


class NoTrainInceptionV3(_FeatureExtractorInceptionV3):
    """Module that never leaves evaluation mode."""

    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the module.

        Args:
            name (str): Name of the module.
            features_list (List[str]): List of features to extract.
            feature_extractor_weights_path (Optional[str]): Path to the weights file.
        """
        assert _TORCH_FIDELITY_AVAILABLE, (
            "NoTrainInceptionV3 module requires that `Torch-fidelity` is installed."
            " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
        )

        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "NoTrainInceptionV3":
        """Force network to always be in evaluation mode."""
        return super().train(False)

    def _torch_fidelity_forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Forward method of inception net.
        Copy of the forward method from this file:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
        with a single line change regarding the casting of `x` in the beginning.
        Corresponding license file (Apache License, Version 2.0):
        https://github.com/toshas/torch-fidelity/blob/master/LICENSE.md

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W)

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors with features extracted from the network.
        """
        vassert(
            torch.is_tensor(x) and x.dtype == torch.uint8,
            "Expecting image as torch.Tensor with dtype=torch.uint8",
        )
        features = {}
        remaining_features = self.features_list.copy()

        x = x.to(self._dtype) if hasattr(self, "_dtype") else x.to(torch.float)
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x, size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), align_corners=False
        )
        x = (x - 128) / 128

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)

        if "64" in remaining_features:
            features["64"] = (
                adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            )
            remaining_features.remove("64")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)

        if "192" in remaining_features:
            features["192"] = (
                adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            )
            remaining_features.remove("192")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        if "768" in remaining_features:
            features["768"] = (
                adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            )
            remaining_features.remove("768")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)

        if "2048" in remaining_features:
            features["2048"] = x
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        if "logits_unbiased" in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features["logits_unbiased"] = x
            remaining_features.remove("logits_unbiased")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        features["logits"] = x
        return tuple(features[a] for a in self.features_list)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of neural network with reshaping of output.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, d)
        """
        out = self._torch_fidelity_forward(x)
        return out[0].reshape(x.shape[0], -1)


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    """
    This function calculates the adjusted version of `Fid Score`_. The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1 (torch.Tensor): Mean of the first Gaussian distribution.
        sigma1 (torch.Tensor): Covariance matrix of the first Gaussian distribution.
        mu2 (torch.Tensor): Mean of the second Gaussian distribution.
        sigma2 (torch.Tensor): Covariance matrix of the second Gaussian distribution.

    Returns:
        torch.Tensor: Frechet Inception Distance between the two distributions.
    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FrechetInceptionDistance(Metric):
    """
    This function calculates the Fréchet inception distance (FID_) which is used to access the quality of generated images.
    .. math::
        FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3
    (`fid ref1`_) features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the
    multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images.
    The metric was originally proposed in `fid ref1`_.

    Using the default feature extraction (Inception v3 using the original weights from `fid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    This metric is known to be unstable in its calculatations, and we recommend for the best results using this metric
    that you calculate using `torch.float64` (default is `torch.float32`) which can be set using the `.set_dtype`
    method of the metric.

    .. note:: using this metrics requires you to have torch 1.9 or higher installed

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fid`` (:class:`~torch.Tensor`): float scalar tensor with mean FID value over samples

    Args:
        feature:
            Either an integer or ``nn.Module``:

            - an integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can be cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    .. note::
        If a custom feature extractor is provided through the `feature` argument it is expected to either have a
        attribute called ``num_features`` that indicates the number of features returned by the forward pass or
        alternatively we will pass through tensor of shape ``(1, 3, 299, 299)`` and dtype ``torch.uint8``` to the
        forward pass and expect a tensor of shape ``(1, num_features)`` as output.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.fid import FrechetInceptionDistance
        >>> fid = FrechetInceptionDistance(feature=64)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> fid.update(imgs_dist1, real=True)
        >>> fid.update(imgs_dist2, real=False)
        >>> fid.compute()
        tensor(12.7202)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    inception: Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[int, Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Frechet Inception Distance metric.

        Args:
            feature (Union[int, Module]): Either an integer or ``nn.Module``. If an
        integer, it indicates the inceptionv3 feature layer to choose. Can be one of the following:
        64, 192, 768, 2048. If an ``nn.Module``, it is a custom feature extractor. Expects that its forward
        method returns an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.
            reset_real_features (bool): Whether to also reset the real features. Since in many cases the real dataset
        does not change, the features can be cached them to avoid recomputing them which is costly. Set this to
        ``False`` if your dataset does not change.
            normalize (bool): Whether to normalize the input images to the feature extractor. If ``True`` images are
        expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if ``False`` images are expected to
        have dtype ``uint8`` and take values in the ``[0, 255]`` range.
            kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        """

        super().__init__(**kwargs)
        assert isinstance(feature, (int, Module)), (
            "Argument `feature` expected to be an int or torch.nn.Module, but got"
            f" {type(feature)}."
        )
        if isinstance(feature, int):
            num_features = feature
            assert _TORCH_FIDELITY_AVAILABLE, (
                "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
            )
            valid_int_input = (64, 192, 768, 2048)
            assert (
                feature in valid_int_input
            ), f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."

            self.inception = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=[str(feature)]
            )

        elif isinstance(feature, Module):
            self.inception = feature
            if hasattr(self.inception, "num_features"):
                num_features = self.inception.num_features
            else:
                dummy_image = torch.randint(0, 255, (1, 3, 299, 299), dtype=torch.uint8)
                num_features = self.inception(dummy_image).shape[-1]

        assert isinstance(reset_real_features, bool), (
            "Argument `reset_real_features` expected to be a bool, but got"
            f" {type(reset_real_features)}."
        )
        self.reset_real_features = reset_real_features
        assert isinstance(normalize, bool), (
            "Argument `normalize` expected to be a bool, but got" f" {type(normalize)}."
        )

        self.normalize = normalize

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, imgs: Tensor, real: bool) -> None:
        """
        Update the state with extracted features.

        Args:
            imgs (torch.Tensor): tensor with images feed to the feature extractor.
            real (bool): bool indicating if ``imgs`` belong to the real or the fake distribution.
        """
        imgs = (imgs * 255).byte() if self.normalize else imgs
        # assets that the input is a tensor of 2D images
        assert imgs.dim() == 4, f"Expected 4D input, got {imgs.dim()}D"
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """
        Calculate FID score based on accumulated extracted features from the two distributions.

        Returns:
            torch.Tensor: Frechet Inception Distance between the two distributions.
        """
        assert (
            self.real_features_num_samples < 2 or self.fake_features_num_samples < 2
        ), "More than one sample is required for both the real and fake distributed to compute FID"

        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """
        Transfer all metric state to specific dtype. Special version of standard `type` method.

        Args:
            dst_type (Union[str, torch.dtype]): The desired dtype of the metric state.

        Returns:
            Metric: The metric with the desired dtype.

        """
        out = super().set_dtype(dst_type)
        if isinstance(out.inception, NoTrainInceptionV3):
            out.inception._dtype = dst_type
        return out
