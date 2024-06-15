from torch import nn
from GANDLF.schedulers import global_schedulers_dict

from typing import Union, Dict


def get_scheduler(scheduler_params: Union[Dict[str, object], str]) -> nn.Module:
    """
    This function gets the scheduler from the parameters.

    Args:
        scheduler_params (Union[Dict[str, object], str]): The dictionary, with key as the scheduler type and value as the parameters for the scheduler.
    or the string containing the name of the scheduler.

    Returns:
        nn.Module : The scheduler object.
    """

    if isinstance(scheduler_params, str):
        scheduler_params = {scheduler_params: {}}

    for scheduler_type, scheduler_params in scheduler_params.items():
        scheduler_type_lower = scheduler_type.lower()

        assert (
            scheduler_type_lower in global_schedulers_dict
        ), f"Scheduler {scheduler_type} not found in the global schedulers dictionary."

        return global_schedulers_dict[scheduler_type_lower](scheduler_params)
