from warnings import warn

from GANDLF.data.augmentation import global_augs_dict

from typing import List


def get_augmentation_transforms(augmentation_params_dict: dict) -> List[object]:
    """
    This function gets the augmentation transformations from the parameters.

    Args:
        augmentation_params_dict (dict): The dictionary containing the parameters for the augmentation.

    Returns:
        List[object]: The list of augmentation to be applied.
    """
    current_augmentations = []

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
