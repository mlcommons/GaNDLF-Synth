from warnings import warn
from gandlf_synth.metrics.synthesis import (
    structural_similarity_index,
    mean_squared_error,
    peak_signal_noise_ratio,
    mean_squared_log_error,
    mean_absolute_error,
    ncc_mean,
    ncc_std,
    ncc_max,
    ncc_min,
    fid,
    lpips,
    ssim_gans,
)

# added all synth metrics from original gandlf, not sure if they will all work tho
global_metrics_dict = {
    "ssim_gan": ssim_gans,
    "fid": fid,
    "lpips": lpips,
    "ncc_mean": ncc_mean,
    "ncc_std": ncc_std,
    "ncc_max": ncc_max,
    "ncc_min": ncc_min,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_log_error": mean_squared_log_error,
    "mean_squared_error": mean_squared_error,
    "peak_signal_noise_ratio": peak_signal_noise_ratio,
    "structural_similarity_index": structural_similarity_index,
}

from typing import List


def get_metrics(metrics_params_dict: dict) -> dict[object]:
    """
    This function gets the metric transformations from the parameters.

    Args:
        metrics_params_dict (dict): The dictionary containing the parameters for the metrics.

    Returns:
        dict[object]: The dict of metrics to be calculated.
    """
    current_metrics = {}

    for metric_type, metric_params in metrics_params_dict.items():
        metric_type_lower = metric_type.lower()

        if metric_type_lower in global_metrics_dict:
            current_metrics[metric_type_lower] = global_metrics_dict[metric_type_lower](
                **metric_params
            )
        else:
            warn(
                f"Metric {metric_type} not found in the global metrics dictionary.",
                UserWarning,
            )
    return current_metrics
