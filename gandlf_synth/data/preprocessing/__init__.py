from warnings import warn

import numpy as np
from torchio.transforms import Resize, Resample

from GANDLF.data.preprocessing import global_preprocessing_dict, Resample_Minimum

from typing import List


def generic_3d_check(patch_size):
    """
    This function reads the value from the configuration and returns an appropriate tuple for torchio to ingest.

    Args:
        patch_size (Union[list, tuple, array]): The generic list/tuple/array to check.

    Returns:
        tuple: The tuple to be ingested by torchio.
    """
    patch_size_new = np.array(patch_size)
    if len(patch_size) == 2:
        patch_size_new = tuple(np.append(np.array(patch_size), 1))

    return patch_size_new


def get_preprocessing_transforms(
    preprocessing_params_dict: dict, train_mode: bool, input_shape: tuple
) -> List[object]:
    """
    This function gets the pre-processing transformations from the parameters.

    Args:
        parameters (dict): The dictionary containing the parameters for the pre-processing.
        train_mode (bool): Whether the data is in train mode or not.
        input_shape (tuple): The input shape of the data.
    Returns:
        List[object]: The list of transformations to be applied.
    """

    # first, we want to do thresholding, followed by clipping, if it is present - required for inference as well
    current_transformations = []
    normalize_to_apply = None
    if not (preprocessing_params_dict is None):
        # go through preprocessing in the order they are specified
        for preprocess in preprocessing_params_dict:
            preprocess_lower = preprocess.lower()
            # special check for resize and resample
            if preprocess_lower == "resize_patch":
                resize_values = generic_3d_check(preprocessing_params_dict[preprocess])
                current_transformations.append(Resize(resize_values))
            elif preprocess_lower == "resample":
                if "resolution" in preprocessing_params_dict[preprocess]:
                    # Need to take a look here
                    resample_values = generic_3d_check(
                        preprocessing_params_dict[preprocess]["resolution"]
                    )
                    current_transformations.append(Resample(resample_values))
            elif preprocess_lower in ["resample_minimum", "resample_min"]:
                if "resolution" in preprocessing_params_dict[preprocess]:
                    resample_values = generic_3d_check(
                        preprocessing_params_dict[preprocess]["resolution"]
                    )
                    current_transformations.append(Resample_Minimum(resample_values))
            # special check for histogram_matching
            elif preprocess_lower == "histogram_matching":
                if preprocessing_params_dict[preprocess] is not False:
                    current_transformations.append(
                        global_preprocessing_dict[preprocess_lower](
                            preprocessing_params_dict[preprocess]
                        )
                    )
            # special check for stain_normalizer
            elif preprocess_lower == "stain_normalizer":
                if normalize_to_apply is None:
                    normalize_to_apply = global_preprocessing_dict[preprocess_lower](
                        preprocessing_params_dict[preprocess]
                    )
            # normalize should be applied at the end
            elif "normalize" in preprocess_lower:
                if normalize_to_apply is None:
                    normalize_to_apply = global_preprocessing_dict[preprocess_lower]
            # preprocessing routines that we only want for training
            elif preprocess_lower in ["crop_external_zero_planes"]:
                if train_mode:
                    current_transformations.append(
                        global_preprocessing_dict["crop_external_zero_planes"](
                            patch_size=input_shape
                        )
                    )
            # everything else is taken in the order passed by user
            elif preprocess_lower in global_preprocessing_dict:
                current_transformations.append(
                    global_preprocessing_dict[preprocess_lower](
                        preprocessing_params_dict[preprocess]
                    )
                )
            else:
                warn(
                    f"Preprocessing {preprocess} not found in the global preprocessing dictionary.",
                    UserWarning,
                )

    # normalization type is applied at the end
    if normalize_to_apply is not None:
        current_transformations.append(normalize_to_apply)

    return current_transformations
