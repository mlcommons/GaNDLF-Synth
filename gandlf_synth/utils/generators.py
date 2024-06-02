import torch


def generate_latent_vector(
    batch_size: int, latent_vector_size: int, dimension: int, device: str
) -> torch.Tensor:
    """
    Creates a latent vector of given size and adjusts the dimensions
    according to the dimension parameter (for 2D or 3D).

    Args:
        batch_size (int): The batch size.
        latent_vector_size (int): The latent vector size.
        dimension (int): The dimension of the images in a given problem.
    Can be `2` for 2D or `3` for 3D.
        device (str): The device to perform computations on.

    Returns:
        latent_vector (torch.Tensor): The latent vector.
    """
    assert dimension in [2, 3], "Dimension should be `2` (2D) or `3` (3D)"
    latent_vector = torch.randn((batch_size, latent_vector_size, 1, 1), device=device)
    if dimension == 3:
        latent_vector = latent_vector.unsqueeze(-1)
    return latent_vector


def get_fixed_latent_vector(
    batch_size: int, latent_vector_size: int, dimension: int, device: str, seed: int
) -> torch.Tensor:
    """
    Function to get the fixed latent vector for inference or validation.
    It always starts with the seed given by user and then re-sets the
    previous RNG state.

    Args:
        batch_size (int): The batch size.
        latent_vector_size (int): The latent vector size.
        dimension (int): The dimension of the images in a given problem.
    Can be `2` for 2D or `3` for 3D.
        device (str): The device to perform computations on.
        seed (int): The seed to use for reproducibility.

    Returns:
        latent_vector (torch.Tensor): The fixed latent vector.
    """
    assert dimension in [2, 3], "Dimension should be 2 (2D) or 3 (3D)"
    current_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    latent_vector = torch.randn((batch_size, latent_vector_size, 1, 1), device=device)
    if dimension == 3:
        latent_vector = latent_vector.unsqueeze(-1)
    torch.set_rng_state(current_rng_state)
    return latent_vector
