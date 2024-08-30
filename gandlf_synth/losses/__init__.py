from copy import deepcopy
from torch.nn import Module
from .generic import CEL, BCELogits, CE, MSE, L1

global_losses_dict = {
    "cel": CEL,
    "l1": L1,
    "bcelogits": BCELogits,
    "ce": CE,
    "mse": MSE,
}


def get_loss(loss_params: dict) -> Module:
    """
    Returns an instance of the specified loss function from the PyTorch `torch.nn` module.

    Args:
        loss_params (dict): Dictionary containing the input parameters for the loss function.

    Returns:
        torch.nn.Module: An instance of the specified loss function.

    """
    # Retrieve the loss function type from the input parameters
    loss_params_copy = deepcopy(loss_params)
    loss_type = loss_params_copy["name"].lower()
    loss_params_copy.pop("name")
    assert (
        loss_type in global_losses_dict
    ), f"Loss function type {loss_type} not found. Please choose from {global_losses_dict.keys()}."

    # Create the loss function instance using the specified type and input parameters
    loss_creator = global_losses_dict[loss_type]
    return loss_creator(loss_params_copy)
