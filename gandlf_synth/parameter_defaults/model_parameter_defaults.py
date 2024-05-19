REQUIRED_MODEL_PARAMETERS = [
    "model_name",
    "n_dimensions",
    "n_channels",
    "losses",
    "optimizers",
]

MODEL_PARAMETER_DEFAULTS = {
    "amp": False,  # automatic mixed precision
    "norm_type": "batch",  # normalization type
    "converter_type": "soft",  # 2d to 3d asc converter type
    "clip_grad": None,  # gradient clipping
    "patience": 0,  # number of epochs to wait for performance improvement
    "schedulers": {},
    "architecture": {},
}
