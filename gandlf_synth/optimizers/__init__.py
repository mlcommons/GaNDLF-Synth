from torch.optim import Optimizer
from .wrap_torch import adam, adamw

from typing import Iterable

global_optimizer_dict = {"adam": adam, "adamw": adamw}


def get_optimizer(model_params: Iterable, optimizer_parameters: dict) -> Optimizer:
    """
    Returns an instance of the specified optimizer from the PyTorch `torch.optim` module.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        params (dict): A dictionary containing the input parameters for the optimizer.
    Returns:
        optimizer (torch.optim.Optimizer): An instance of the specified optimizer.

    """
    # Retrieve the optimizer type from the input parameters
    optimizer_type = optimizer_parameters["name"]
    optimizer_parameters.pop("name")
    assert (
        optimizer_type in global_optimizer_dict
    ), f"Optimizer type {optimizer_type} not found. Please choose from {global_optimizer_dict.keys()}."
    # Create the optimizer instance using the specified type and input parameters
    optimizer_creator = global_optimizer_dict[optimizer_type]
    return optimizer_creator(model_params, optimizer_parameters)
