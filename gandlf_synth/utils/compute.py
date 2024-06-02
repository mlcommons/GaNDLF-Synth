import torch
from GANDLF.grad_clipping.grad_scaler import GradScaler, model_parameters_exclude_head
from GANDLF.grad_clipping.clip_gradients import dispatch_clip_grad_

from typing import Optional


def backward_pass(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    amp: Optional[bool] = False,
    clip_grad: Optional[float] = None,
    clip_mode: Optional[str] = "norm",
) -> None:
    """
    Function to perform the backward pass for a single batch.

    Args:
        loss (torch.Tensor): The loss to backpropagate.
        optimizer (torch.optim.Optimizer): The optimizer to use for backpropagation.
        model (torch.nn.Module): The model to backpropagate through.
        amp (Optional[bool], optional): Whether to use automatic mixed precision. Defaults to False.
        clip_grad (Optional[float], optional): The clipping value/factor/norm, mode dependent. Defaults to None.
        clip_mode (Optional[str], optional): The mode of clipping. Defaults to "norm".
    """
    nan_loss: torch.Tensor = torch.isnan(loss)
    second_order: bool = (
        hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    )
    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if amp:
        scaler: GradScaler = GradScaler()
        with torch.cuda.amp.autocast():
            # if loss is nan, don't backprop and don't step optimizer
            if not nan_loss:
                scaler(
                    loss=loss,
                    optimizer=optimizer,
                    clip_grad=clip_grad,
                    clip_mode=clip_mode,
                    parameters=model_parameters_exclude_head(
                        model, clip_mode=clip_mode
                    ),
                    create_graph=second_order,
                )
    else:
        if not nan_loss:
            loss.backward(create_graph=second_order)
            if clip_grad is not None:
                dispatch_clip_grad_(
                    parameters=model_parameters_exclude_head(
                        model, clip_mode=clip_mode
                    ),
                    value=clip_grad,
                    mode=clip_mode,
                )
