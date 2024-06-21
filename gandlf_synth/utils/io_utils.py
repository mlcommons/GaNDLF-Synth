import torch
import numpy as np
import SimpleITK as sitk


EXTENSION_MAP = {"rad": ".nii.gz", "histo": ".tiff"}


# TODO check if this will handle 2d and 3d data properly
def save_single_image(
    image: np.ndarray, image_path: str, modality: str, dimensionality: int
):
    """Save the image to the given path. Uses proper extension based on the modality.

    Args:
        image (np.ndarray): The image to save.
        image_path (str): The path to save the image.
        modality (str): The modality of the image.
        dimensionality (int): The dimensionality of the image.
    """
    extension = EXTENSION_MAP[modality]
    image_path = image_path + extension
    image_copied = image.copy().squeeze()
    is_vector = dimensionality == 2
    sitk_image = sitk.GetImageFromArray(image_copied, isVector=is_vector)
    sitk.WriteImage(sitk_image, image_path)
