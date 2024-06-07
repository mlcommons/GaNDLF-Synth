from typing import Optional
import SimpleITK as sitk
import PIL.Image
import numpy as np
import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    MeanSquaredError,
    MeanSquaredLogError,
    MeanAbsoluteError,
    PeakSignalNoiseRatio,
)
from GANDLF.utils import get_image_from_tensor
import warnings
from typing import Any, Dict, Tuple
from gandlf_synth.metrics.utils.lpip import LPIPSGandlf
from gandlf_synth.metrics.utils.fid import FrechetInceptionDistance


def structural_similarity_index(
    prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the structural similarity index between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
        mask (Optional[torch.Tensor], optional): The mask tensor. Defaults to None.

    Returns:
        torch.Tensor: The structural similarity index.
    """
    ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
    _, ssim_idx_full_image = ssim(preds=prediction, target=target)
    mask = torch.ones_like(ssim_idx_full_image) if mask is None else mask
    try:
        ssim_idx = ssim_idx_full_image[mask]
    except Exception as e:
        print(f"Error: {e}")
        if len(ssim_idx_full_image.shape) == 0:
            ssim_idx = torch.ones_like(mask) * ssim_idx_full_image
    return ssim_idx.mean()


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
    """
    mse = MeanSquaredError()
    return mse(preds=prediction, target=target)


def peak_signal_noise_ratio(
    target: torch.Tensor,
    prediction: torch.Tensor,
    data_range: Optional[tuple] = None,
    epsilon: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the peak signal to noise ratio between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.
        data_range (Optional[tuple], optional): The data range. Defaults to None.
        epsilon (Optional[float], optional): The epsilon value. Defaults to None.

    Returns:
        torch.Tensor: The peak signal to noise ratio.
    """
    if epsilon == None:
        psnr = (
            PeakSignalNoiseRatio()
            if data_range == None
            else PeakSignalNoiseRatio(data_range=data_range[1] - data_range[0])
        )
        return psnr(preds=prediction, target=target)
    else:  # implementation of PSNR that does not give 'inf'/'nan' when 'mse==0'
        mse = mean_squared_error(target, prediction)
        if data_range == None:  # compute data_range like torchmetrics if not given
            min_v = (
                0 if torch.min(target) > 0 else torch.min(target)
            )  # look at this line
            max_v = torch.max(target)
        else:
            min_v, max_v = data_range
        return 10.0 * torch.log10(((max_v - min_v) ** 2) / (mse + epsilon))


def mean_squared_log_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the mean squared log error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The mean squared log error.
    """
    mle = MeanSquaredLogError()
    return mle(preds=prediction, target=target)


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean absolute error between the target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The mean absolute error.
    """
    mae = MeanAbsoluteError()
    return mae(preds=prediction, target=target)


def _get_ncc_image(prediction: torch.Tensor, target: torch.Tensor) -> sitk.Image:
    """
    Computes normalized cross correlation image between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        sitk.Image: The normalized cross correlation image.
    """

    def __convert_to_grayscale(image: sitk.Image) -> sitk.Image:
        """
        Helper function to convert image to grayscale.

        Args:
            image (sitk.Image): The image to convert.

        Returns:
            sitk.Image: The converted image.
        """
        if "vector" in image.GetPixelIDTypeAsString().lower():
            temp_array = sitk.GetArrayFromImage(image)
            image_pil = PIL.Image.fromarray(
                np.moveaxis(temp_array[0, ...], 0, 2).astype(np.uint8)
            )
            image_pil_gray = image_pil.convert("L")
            return sitk.GetImageFromArray(image_pil_gray)
        else:
            return image

    target_image = __convert_to_grayscale(get_image_from_tensor(target))
    pred_image = __convert_to_grayscale(get_image_from_tensor(prediction))
    correlation_filter = sitk.FFTNormalizedCorrelationImageFilter()
    return correlation_filter.Execute(target_image, pred_image)


def ncc_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation mean between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation mean.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMean()


def ncc_std(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation standard deviation between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation standard deviation.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetSigma()


def ncc_max(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation maximum between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation maximum.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMaximum()


def ncc_min(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes normalized cross correlation minimum between target and prediction.

    Args:
        prediction (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The normalized cross correlation minimum.
    """
    stats_filter = sitk.StatisticsImageFilter()
    corr_image = _get_ncc_image(target, prediction)
    stats_filter.Execute(corr_image)
    return stats_filter.GetMinimum()


def _structural_similarity_index_measure(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the SSIM between the generated images and the real
    images. Except for the params specified below, the rest of the params are
    default from torchmetrics. Works both for 2D and 3D images.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The SSIM score.
    """

    def _get_reduction(params: Dict[str, Any]) -> str:
        """
        This function returns the reduction type from config.

        Args:
            params (dict): The parameter dictionary containing training and data information.

        Returns:
            str: The reduction type.
        """
        # check if metrics have config
        if "metrics_config" in params:
            # check if ssim has config
            if "ssim" in params["metrics_config"]:
                # check if reduction is present
                if "reduction" in params["metrics_config"]["ssim"]:
                    return params["metrics_config"]["ssim"]["reduction"]
        return "elementwise_mean"

    reduction = _get_reduction(params)
    if reduction not in ["elementwise_mean", "sum"]:
        warnings.warn(
            f"Reduction type {reduction} not supported. Defaulting to "
            "elementwise_mean.",
            UserWarning,
        )
        reduction = "elementwise_mean"

    if params["model"]["dimension"] == 2:
        real_images = real_images.squeeze(-1)
    ssim = StructuralSimilarityIndexMeasure(reduction=reduction)  # type: ignore
    return ssim(generated_images, real_images)


def _ferechet_inception_distance(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the FID between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The FID score.
    """
    assert (generated_images.dim() == 4) and (real_images.dim() == 4), (
        "FID is only supported for 2D images. "
        "The input images should be of shape (batch_size, channels, height, width)."
        f"Got generated_images of shape {generated_images.shape} and real_images of shape {real_images.shape}"
    )

    def _get_features_size(params: Dict[str, Any]) -> int:
        """
        This function returns the feature size for FID from config.

        Args:
            params (dict): The parameter dictionary containing training and data information.

        Returns:
            int: The feature size.
        """
        # check if metrics have config
        if "metrics_config" in params:
            # check if fid has config
            if "fid" in params["metrics_config"]:
                # check if features_size is present
                if "features_size" in params["metrics_config"]["fid"]:
                    return params["metrics_config"]["fid"]["features_size"]
        return 2048

    assert params["model"]["dimension"] == 2, "FID is only supported for 2D images"

    assert params["batch_size"] > 1, "FID is not supported for batch size 1"
    real_images = real_images.squeeze(-1)

    features_size = _get_features_size(params)
    fid_metric = FrechetInceptionDistance(feature=features_size, normalize=True)
    n_input_channels = params["model"]["num_channels"]
    if n_input_channels == 1:
        # need manual patching for single channel data
        fid_metric.get_submodule("inception")._modules["Conv2d_1a_3x3"]._modules[
            "conv"
        ] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # check input dtype
    if generated_images.dtype != torch.float32:
        generated_images = generated_images.float()
    if real_images.dtype != torch.float32:
        real_images = real_images.float()
    if generated_images.max() > 1:
        warnings.warn(
            "Input generated images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation.",
            UserWarning,
        )
        generated_images = generated_images / 255.0
    if real_images.max() > 1:
        warnings.warn(
            "Input real images are not in [0, 1] range. "
            "This may lead to incorrect results. "
            "FID expects input images to be in [0, 1] range."
            "Dividing the images by 255 for metric calculation.",
            UserWarning,
        )
        real_images = real_images / 255.0
    if generated_images.shape[0] == 1 or real_images.shape[0] == 1:
        warnings.warn(
            "FID is not supported for batch size 1. "
            "The metric will not be computed for this batch.",
            UserWarning,
        )
        return torch.tensor([0.0])

    fid_metric.update(generated_images, real=False)
    fid_metric.update(real_images, real=True)
    metric_value = fid_metric.compute()
    return metric_value


def _learned_perceptual_image_patch_similarity(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the LPIPS between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.
        n_dim (int): The number of dimensions.
        net_type (Literal["alex", "squeeze", "vgg"], optional): The network type.
    Defaults to "squeeze".
        reduction (Literal["mean", "sum"], optional): The reduction type.
    Defaults to "mean".
        converter_type (Literal["soft", "acs", "conv3d], optional): The converter
    type from ACS. Defaults to "soft".

    Returns:
        torch.Tensor: The LPIP score.
    """

    def _get_metric_params(params: Dict[str, Any]) -> Tuple[int, int, str, str, str]:
        """
        This function returns the metric parameters from config.

        Args:
            params (dict): The parameter dictionary containing training and data information.

        Returns:
            Tuple[int, int, str, str, str]: The metric parameters,
        namely n_input_channels, n_dim, net_type, reduction, converter_type.
        """
        n_input_channels = params["model"]["num_channels"]
        n_dim = params["model"]["dimension"]
        if "metrics_config" in params:
            if "lpips" in params["metrics_config"]:
                net_type = (
                    params["metrics_config"]["lpips"]["net_type"]
                    if "net_type" in params["metrics_config"]["lpips"]
                    else "squeeze"
                )
                reduction = (
                    params["metrics_config"]["lpips"]["reduction"]
                    if "reduction" in params["metrics_config"]["lpips"]
                    else "mean"
                )
                converter_type = (
                    params["metrics_config"]["lpips"]["converter_type"]
                    if "converter_type" in params["metrics_config"]["lpips"]
                    else "soft"
                )
                return (n_input_channels, n_dim, net_type, reduction, converter_type)
        return n_input_channels, n_dim, "squeeze", "mean", "soft"

    def _ensure_proper_scale_and_dtype(images: torch.Tensor) -> torch.Tensor:
        """
        This function ensures that the input images are in the correct scale
        and dtype for the metric calculation.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The scaled and dtype corrected images.
        """
        if images.dtype != torch.float32:
            images = images.float()
        # if images are in [-1, 1] range, scale to [0, 1]
        if not torch.all((images >= 0) & (images <= 1)):
            warnings.warn(
                "Input images are not in [0, 1] range. "
                "This may lead to incorrect results. "
                "LPIPS expects input images to be in [0, 1] range."
                "Performing min-max scaling for metric calculation.",
                UserWarning,
            )
            images = (images - images.min()) / (images.max() - images.min())
        return images

    (n_input_channels, n_dim, net_type, reduction, converter_type) = _get_metric_params(
        params
    )
    if n_dim == 2:
        real_images = real_images.squeeze(-1)
    else:
        warnings.warn(
            "LPIPS was originally designed for 2D data. "
            "Currently, the entire network will be modified to accept 3D "
            "with ACS converter. (https://arxiv.org/abs/1911.10477)"
            "Results need to be interpreted with caution.",
            UserWarning,
        )
    if n_input_channels == 1:
        warnings.warn(
            "LPIPS was designed for 3-channel images. "
            "Currently, the input layer will be modified to accept single "
            "channel images. Results need to be interpreted with caution.",
            UserWarning,
        )
    lpips_metric = LPIPSGandlf(
        net_type=net_type,  # type: ignore
        normalize=True,
        reduction=reduction,  # type: ignore
        n_dim=n_dim,
        n_channels=n_input_channels,
        converter_type=converter_type,  # type: ignore
    )

    # check input dtype
    generated_images = _ensure_proper_scale_and_dtype(generated_images)
    real_images = _ensure_proper_scale_and_dtype(real_images)

    return lpips_metric(generated_images, real_images)


def fid(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the FID between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        n_input_channels (int): The number of input channels.

    Returns:
        torch.Tensor: The FID score.
    """
    return _ferechet_inception_distance(generated_images, real_images, params)


def lpips(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the LPIPS between the generated images and the
    real images. Except for the params specified below, the rest of the params
    are default from torchmetrics.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The LPIP score.
    """
    return _learned_perceptual_image_patch_similarity(
        generated_images, real_images, params
    )


def ssim_gans(
    generated_images: torch.Tensor, real_images: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """
    This function computes the SSIM between the generated images and the real
    images. Except for the params specified below, the rest of the params are
    default from torchmetrics.

    Args:
        generated_images (torch.Tensor): The generated images.
        real_images (torch.Tensor): The real images.
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        torch.Tensor: The SSIM score.
    """
    return _structural_similarity_index_measure(generated_images, real_images, params)
