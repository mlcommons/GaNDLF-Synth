from .synthesis import (
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
