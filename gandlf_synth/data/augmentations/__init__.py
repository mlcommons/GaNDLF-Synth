from warnings import warn

from GANDLF.data.augmentation import global_augs_dict

from typing import List, Union, Dict, Callable


def get_augmentation_transforms(
    augmentation_params_dict: Union[Dict[str, object], List[str]]
) -> List[Callable]:
    """
    This function gets the augmentation transformations from the parameters.

    Args:
        augmentation_params_dict (dict): The dictionary containing the parameters for the augmentation.

    Returns:
        List[Callable]: The list of augmentation to be applied.
    """
    current_augmentations = []

    # Check if user specified some augmentations without extra params
    if isinstance(augmentation_params_dict, list):
        for n, augmentation_type in enumerate(augmentation_params_dict):
            if isinstance(augmentation_type, dict):
                continue
            else:
                augmentation_params_dict[n] = {augmentation_type: {}}

    for augmentation_type, augmentation_params in augmentation_params_dict.items():
        augmentation_type_lower = augmentation_type.lower()

        if augmentation_type_lower in global_augs_dict:
            current_augmentations.append(
                global_augs_dict[augmentation_type_lower](**augmentation_params)
            )
        else:
            warn(
                f"Augmentation {augmentation_type} not found in the global augmentation dictionary.",
                UserWarning,
            )
    return current_augmentations
