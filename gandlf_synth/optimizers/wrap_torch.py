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

    return Adam(model_params, **optimizer_parameters)


def adamw(model_params: Iterable, optimizer_parameters: dict) -> Optimizer:
    """
    Creates an AdamW optimizer from the PyTorch `torch.optim` module using the input parameters.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        optimizer_parameters (dict): A dictionary containing the input parameters for the optimizer.
    Returns:
        optimizer (torch.optim.AdamW): AdamW optimizer.
    """

    return AdamW(model_params, **optimizer_parameters)
