import os
import logging
from torchio.transforms import Compose

from GANDLF.data.augmentation import get_augmentation_transforms
from gandlf_synth.data.preprocessing import get_preprocessing_transforms
from gandlf_synth.data.postprocessing import get_postprocessing_transforms
from typing import List, Optional, Callable, Union, Callable


def prepare_logger(logger_name: str, model_dir_path: str) -> logging.Logger:
    """
    Prepare the logger.

    Args:
        logger_name (str): The name of the logger.
        model_dir_path (str): The path to the model directory.

    Returns:
        logging.Logger: The logger.
    """
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
    filehandler = logging.FileHandler(
        os.path.join(model_dir_path, f"{logger_name}.log")
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    filehandler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(filehandler)
    return logger


def prepare_postprocessing_transforms(global_config: dict) -> List[Callable]:
    """
    Prepare the postprocessing transforms from config.

    Args:
        global_config (dict): The global config.

    Returns:
        List[Callable]: The list of postprocessing transforms.
    """
    postprocessing_transforms = None
    postprocessing_config = global_config.get("data_postprocessing")
    if postprocessing_config is not None:
        postprocessing_transforms = get_postprocessing_transforms(postprocessing_config)
    return postprocessing_transforms


def prepare_transforms(
    preprocessing_config: Union[dict, None],
    augmentations_config: Union[dict, None],
    mode: str,
    input_shape: tuple,
) -> Compose:
    """
    Prepare the transforms for either training, validation, testing or inference datasets.

    Args:
        preprocessing_config (dict): The preprocessing configuration.
        augmentations_config (dict): The augmentations configuration.
        mode (str): The mode for which the transforms are being prepared (train, val, test).
        input_shape (tuple): The input shape of the data.
    """
    assert mode in [
        "train",
        "val",
        "test",
        "inference",
    ], "Mode must be one of 'train', 'val', 'test' or 'inference'."
    transforms_list = []
    preprocessing_operations = None
    augmentation_operations = None
    train_mode = True if mode == "train" else False
    if preprocessing_config is not None:
        preprocessing_operations = preprocessing_config.get(mode)
    if augmentations_config is not None:
        augmentation_operations = augmentations_config.get(mode)
    if preprocessing_operations is not None:
        preprocessing_transforms = get_preprocessing_transforms(
            preprocessing_operations, train_mode, input_shape
        )
        transforms_list.extend(preprocessing_transforms)
    # as in Gandlf, we will use augmentations only in training mode
    if augmentation_operations is not None and train_mode:
        augmentation_transforms = get_augmentation_transforms(augmentation_operations)
        transforms_list.extend(augmentation_transforms)
    if len(transforms_list) > 0:
        return Compose(transforms_list)


def determine_checkpoint_to_load(
    model_dir: str, custom_checkpoint_path: Optional[str]
) -> Union[str, None]:
    """
    Determine the checkpoint to load for the inference process. Used in training
    and validation managers. The checkpoint resolution order is as follows:
    1. Custom checkpoint path.
    2. Best checkpoint path.
    3. Last checkpoint path.
    If none of the above are found, the function will return None.

    Args:
        model_dir (str): The model directory path.
        custom_checkpoint_path (Optional[str]): The custom checkpoint path.

    Returns:
        Union[str, None]: The checkpoint path to load.
    """
    if custom_checkpoint_path is not None:
        return custom_checkpoint_path
    best_checkpoint_path = os.path.join(model_dir, "checkpoints", "best.ckpt")
    if os.path.exists(best_checkpoint_path):
        return best_checkpoint_path
    last_checkpoint_path = os.path.join(model_dir, "checkpoints", "last.ckpt")
    if os.path.exists(last_checkpoint_path):
        return last_checkpoint_path
