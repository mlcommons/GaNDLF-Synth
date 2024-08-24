REQUIRED_MODEL_PARAMETERS = [
    "model_name",
    "n_dimensions",
    "n_channels",
    "losses",
    "optimizers",
    "tensor_shape",
]

MODEL_PARAMETER_DEFAULTS = {
    "norm_type": "batch",  # normalization type
    "converter_type": "soft",  # 2d to 3d asc converter type
    "accumulate_grad_batches": 1,  # number of batches to accumulate gradients
    "gradient_clip_val": None,  # gradient clipping
    "gradient_clip_algorithm": "norm",  # gradient clipping mode, either "norm" or "value"
    "patience": 0,  # number of epochs to wait for performance improvement
    "labeling_paradigm": "unlabeled",  # labeling paradigm
    "inference_parameters": {},  # inference parameters
    "schedulers": None,
    "architecture": {},
}
