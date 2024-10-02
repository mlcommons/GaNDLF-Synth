from torch import optim
from GANDLF.schedulers import global_schedulers_dict

from typing import Union, Dict


def get_scheduler(
    optimizer: optim.Optimizer, scheduler_params: Union[Dict[str, object], str]
) -> optim.lr_scheduler._LRScheduler:
    """
    This function gets the scheduler from the parameters. It serves also
    as a connector to the gandlf-core API for the schedulers.
    Args:
        optimizer (optim.Optimizer): The optimizer object.
        scheduler_params (Union[Dict[str, object], str]): The dictionary, with key as the scheduler type and value as the parameters for the scheduler.
    or the string containing the name of the scheduler.

    Returns:
        optim.lr_scheduler._LRScheduler: The scheduler object.
    """

    if isinstance(scheduler_params, str):
        scheduler_params = {"scheduler": {"type": scheduler_params}}
    else:
        scheduler_params = {"scheduler": scheduler_params}
    scheduler_params["optimizer_object"] = optimizer
    scheduler_type: str = scheduler_params["scheduler"]["type"].lower()
    assert (
        scheduler_type in global_schedulers_dict
    ), f"Scheduler {scheduler_type} not found in the global schedulers dictionary."

    return global_schedulers_dict[scheduler_type](scheduler_params)
