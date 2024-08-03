import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchio.transforms import Compose

from GANDLF.data.augmentation import get_augmentation_transforms
from gandlf_synth.data.preprocessing import get_preprocessing_transforms
from gandlf_synth.data.postprocessing import get_postprocessing_transforms
from typing import List, Tuple, Callable, Union, Callable


def prepare_logger(logger_name: str) -> logging.Logger:
    """
    Prepare the logger.

    Args:
        logger_name (str): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
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


def assert_input_correctness(
    configured_input_shape: Tuple[int],
    configured_n_channels: int,
    batch_idx: int,
    batch: object,
):
    """
    Assert the correctness of the input shape in a given data batch.

    Args:
        configured_input_shape (Tuple[int]): The configured input shape.
        configured_n_channels (int): The configured number of channels.
        batch_idx (int): The index of the batch.
        batch (object): The data batch.
    """

    expected_input_shape = [configured_n_channels] + configured_input_shape
    # maybe in  the upcoming PRs we should consider some dict-like
    # structure returned by the dataloader? So we can access the data
    # by keywords, like batch["image"] or batch["label"] instead of
    # indices
    batch_image_shape = list(batch[0].shape)
    assert (
        batch_image_shape == expected_input_shape
    ), f"Batch {batch_idx} has incorrect shape. Expected: {expected_input_shape}, got: {batch_image_shape}"


def prepare_progress_bar(
    dataloader: DataLoader, total_size: int, description: str, **kwargs
):
    """
    Prepare the progress bar for the dataloader.

    Args:
        dataloader (DataLoader): The dataloader.
        total_size (int): The total size of the dataloader.
        description (str): The description of the progress bar.
        **kwargs: Additional keyword arguments for tqdm.

    Returns:
        progress_bar (tqdm.tqdm): The progress bar.
    """
    return tqdm(enumerate(dataloader), total=total_size, desc=description, **kwargs)
