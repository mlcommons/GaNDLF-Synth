import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss, Module

from typing import Optional


def get_weights(loss_params: dict) -> Optional[torch.Tensor]:
    """
    Helper function to get the weights for the loss function.

    Args:
        loss_params (dict): Dictionary containing the loss parameters.

    Returns:
        torch.Tensor: Weights for the loss function.
    """

    weights = None
    penalty_weights = loss_params.get("penalty_weights")
    if penalty_weights is not None:
        weights = torch.FloatTensor(list(penalty_weights.values()))
    return weights


# TODO Need to check if for BCELogits the weights are properly applied
def BCELogits(loss_params: dict) -> Module:
    """
    Binary cross entropy loss with logits.

    Args:
        loss_params (dict): Dictionary containing the loss parameters.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.
    """

    return BCEWithLogitsLoss(**loss_params)


def CE(loss_params: dict) -> Module:
    """
    Binary cross entropy loss.

    Args:
        loss_params (dict): Dictionary containing the loss parameters.

    Returns:
        torch.Tensor: Binary cross entropy loss tensor.

    """
    weights = get_weights(loss_params)
    return BCELoss(weight=weights, **loss_params)


def CEL(loss_params: dict) -> Module:
    """
    Cross entropy loss with optional class weights.

    Args:
        loss_params (dict): Dictionary containing the loss parameters.

    Returns:
        torch.Tensor: Cross entropy loss tensor.
    """
    weights = get_weights(loss_params)
    return CrossEntropyLoss(weight=weights, **loss_params)
