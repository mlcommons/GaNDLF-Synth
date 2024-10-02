import torch
from copy import deepcopy
from torch.nn import Module
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss

global_losses_dict = {
    "cel": CrossEntropyLoss,
    "l1": L1Loss,
    "bce": BCELoss,
    "bcelogits": BCEWithLogitsLoss,
    "mse": MSELoss,
}

WEIGHT_KEYS = ["weight", "pos_weight"]


def get_loss(loss_params: dict) -> Module:
    """
    Returns an instance of the specified loss function from the PyTorch `torch.nn` module.

    Args:
        loss_params (dict): Dictionary containing the input parameters for the loss function.

    Returns:
        torch.nn.Module: An instance of the specified loss function.
    """
    # Retrieve the loss function type from the input parameters
    # We operate on a deepcopy to avoid modifying the original input parameters
    loss_params_copy = deepcopy(loss_params)
    loss_type = loss_params_copy["name"].lower()
    loss_params_copy.pop("name")
    loss_params_copy = convert_weight_parameters(loss_params_copy)
    assert (
        loss_type in global_losses_dict
    ), f"Loss function type {loss_type} not found. Please choose from {global_losses_dict.keys()}."

    return global_losses_dict[loss_type](**loss_params_copy)


def convert_weight_parameters(loss_params: dict) -> dict:
    """
    Converts the weight parameters to torch tensors.
    Some loss functions allow for specifying weights for different classes or samples,
    we therefore must ensure that these weights are torch tensors.

    Args:
        loss_params (dict): Dictionary containing the input parameters for the loss function.

    Returns:
        dict: Dictionary containing the input parameters with the weight parameters converted to torch tensors.
    """
    # check if any of the weight parameters are present in the loss_params provided
    occuring_weight_params = [k for k in WEIGHT_KEYS if k in loss_params.keys()]
    if occuring_weight_params:
        for keyword in occuring_weight_params:
            loss_params[keyword] = torch.tensor(
                loss_params[keyword], dtype=torch.float32
            )
    return loss_params
