from warnings import warn

from GANDLF.data.post_process import global_postprocessing_dict

from typing import List, Union, Dict, Callable


def get_postprocessing_transforms(
    postprocessing_params: Union[Dict[str, object], List[str]]
) -> List[Callable]:
    """
    This function gets the postprocessing transformations from the parameters.

    Args:
        postprocessing_params (dict): The dictionary containing the parameters for the postprocessing.

    Returns:
        List[Callable] : The list of postprocessing operations to apply.
    """

    current_postprocessing_ops = []

    # Check if user specified some postprocessing without extra params

    if isinstance(postprocessing_params, list):
        for n, postprocessing_type in enumerate(postprocessing_params):
            if isinstance(postprocessing_type, dict):
                continue
            else:
                postprocessing_params[n] = {postprocessing_type: {}}

    for postprocessing_type, postprocessing_params in postprocessing_params.items():
        postprocessing_type_lower = postprocessing_type.lower()

        if postprocessing_type_lower in global_postprocessing_dict:
            current_postprocessing_ops.append(
                global_postprocessing_dict[postprocessing_type_lower](
                    **postprocessing_params
                )
            )
        else:
            warn(
                f"Postprocessing {postprocessing_type} not found in the global postprocessing dictionary.",
                UserWarning,
            )
    return current_postprocessing_ops
