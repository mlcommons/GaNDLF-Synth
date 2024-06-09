from warnings import warn
from GANDLF.metrics.synthesis import (
    structural_similarity_index,
    mean_squared_error,
    peak_signal_noise_ratio,
    mean_squared_log_error,
    mean_absolute_error,
    ncc_mean,
    ncc_std,
    ncc_max,
    ncc_min,
)

# added all synth metrics from original gandlf, not sure if they will all work tho
global_metrics_dict = {
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

from typing import List, Union, Dict, Callable


def get_metrics(
    metrics_params: Union[Dict[str, object], List[str]]
) -> Dict[str, Callable]:
    """
    This function gets the metric transformations from the parameters.

    Args:

        metrics_params (Union[Dict[str, object], List[str]]): The metrics parameters.
    Can be a list of metric names or a dictionary of metric names and their parameters.
    Returns:
        dict[object]: The dict of metrics to be calculated.
    """
    current_metrics = {}

    # Converting the list of metrics to a dictionary format.
    if isinstance(metrics_params_dict, list):
        converted_metrics_params = {}
        for metric_type in metrics_params_dict:
            if isinstance(metric_type, dict):
                # case in which user specified some metrics with parameters along with some metrics without parameters
                converted_metrics_params[metric_type.keys()] = metric_type.values()
            else:
                converted_metrics_params[metric_type] = {}
        metrics_params = converted_metrics_params
    # TODO: Currently most of those transforms DO NOT support additional params.
    # It is a good idea to add support for additional parameters in the future.
    # Espeially that things like SSIM require additional parameters.
    for metric_type, metric_params in metrics_params.items():
        metric_type_lower = metric_type.lower()

        if metric_type_lower in global_metrics_dict:
            current_metrics[metric_type_lower] = global_metrics_dict[metric_type_lower]
        else:
            warn(
                f"Metric {metric_type} not found in the global metrics dictionary.",
                UserWarning,
            )
    return current_metrics
