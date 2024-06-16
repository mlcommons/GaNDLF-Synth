from copy import deepcopy
from torch.optim import Optimizer
from GANDLF.optimizers import get_optimizer as get_optimizer_gandlf

from typing import Iterable, Tuple


def parse_optimizer_parameters_to_gandlf_format(
    model_params: Iterable, optimizer_parameters: dict
) -> Tuple[str, dict]:
    """
    This function parses the optimizer parameters to the format required by GANDLF.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        optimizer_parameters (dict): A dictionary containing the input parameters for the optimizer.

    Returns:
        new_full_dict (dict): A dictionary containing the model parameters and the optimizer parameters in the format required by GANDLF.
    """
    parsed_optimizer_parameters = {}
    # Do not alter the original dictionary
    new_params_dict = deepcopy(optimizer_parameters)
    optimizer_name = new_params_dict.pop("name")
    # Converting the `lr` key to `learning_rate`, for GANDLF compatibility
    parsed_optimizer_parameters["learning_rate"] = new_params_dict.pop("lr")
    parsed_optimizer_parameters["model_parameters"] = model_params
    parsed_optimizer_parameters["optimizer"] = new_params_dict
    parsed_optimizer_parameters["optimizer"]["type"] = optimizer_name
    return parsed_optimizer_parameters


def get_optimizer(model_params: Iterable, optimizer_parameters: dict) -> Optimizer:
    """
    Returns an instance of the specified optimizer from the PyTorch `torch.optim` module.

    Args:
        model_params (iterable): An iterable containing the model parameters to optimize.
        params (dict): A dictionary containing the input parameters for the optimizer.
    Returns:
        optimizer (torch.optim.Optimizer): An instance of the specified optimizer.

    """
    parsed_optimizer_parameters = parse_optimizer_parameters_to_gandlf_format(
        model_params=model_params, optimizer_parameters=optimizer_parameters
    )
    return get_optimizer_gandlf(parsed_optimizer_parameters)
