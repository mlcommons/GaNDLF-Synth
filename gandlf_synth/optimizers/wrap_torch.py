from torch.optim import (
    Optimizer,
    SGD,
    ASGD,
    Rprop,
    Adam,
    AdamW,
    # SparseAdam,
    Adamax,
    Adadelta,
    Adagrad,
    RMSprop,
    RAdam,
)

from typing import Iterable


def adam(model_params: Iterable, optimizer_parameters: dict) -> Optimizer:
    """
    Creates an Adam optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        optimizer_parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        optimizer (torch.optim.Adam): An Adam optimizer.
    """

    return Adam(
        model_params,
        lr=optimizer_parameters.get("learning_rate"),
        betas=optimizer_parameters.get("betas", (0.9, 0.999)),
        weight_decay=optimizer_parameters.get("weight_decay", 0.00005),
        eps=optimizer_parameters.get("eps", 1e-8),
        amsgrad=optimizer_parameters.get("amsgrad", False),
    )


def adamw(model_params: Iterable, optimizer_parameters: dict) -> Optimizer:
    """
    Creates an AdamW optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        optimizer_parameters (dict): A dictionary containing the input parameters for the optimizer.
    Returns:
        optimizer (torch.optim.AdamW): AdamW optimizer.
    """

    return AdamW(
        model_params,
        lr=optimizer_parameters.get("learning_rate"),
        betas=optimizer_parameters.get("betas", (0.9, 0.999)),
        weight_decay=optimizer_parameters.get("weight_decay", 0.00005),
        eps=optimizer_parameters.get("eps", 1e-8),
        amsgrad=optimizer_parameters.get("amsgrad", False),
    )
