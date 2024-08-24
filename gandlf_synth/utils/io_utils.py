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
    is_vector = dimensionality == 2
    sitk_image = sitk.GetImageFromArray(image.squeeze(), isVector=is_vector)
    sitk.WriteImage(sitk_image, image_path)


def prepare_images_for_saving(
    generated_images: torch.Tensor, n_dimensions: int
) -> np.ndarray:
    """
    Prepare the generated images for saving, permuting the dimensions and
    converting them to numpy arrays for saving with SimpleITK.

    Args:
        generated_images (torch.Tensor): The generated images.
        n_dimensions (int): The number of dimensions of the images.
    Returns:
        np.ndarray: The generated images prepared for saving.
    """
    if n_dimensions == 2:
        return generated_images.permute(0, 2, 3, 1).cpu().numpy()
    elif n_dimensions == 3:
        return generated_images.permute(0, 2, 3, 4, 1).cpu().numpy()
