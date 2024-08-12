import os
import logging
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from gandlf_synth.utils.compute import ensure_device_placement
from gandlf_synth.utils.io_utils import save_single_image
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.models.configs.config_abc import AbstractModelConfig

from typing import Tuple, Type, Dict, Any, Optional


def determine_n_batches(n_images_to_generate: int, batch_size: int) -> Tuple[int, int]:
    """
    Determine the number of batches needed to generate the images.

    Args:
        n_images_to_generate (int): The number of images to generate.
        batch_size (int): The batch size.

    Returns:
        Tuple[int, int]: The number of batches and the remainder.
    """
    n_batches = n_images_to_generate // batch_size
    remainder = n_images_to_generate % batch_size
    if remainder > 0:
        n_batches += 1
    return n_batches, remainder


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


class InferenceStrategy(ABC):
    """
    Class to manage the inference strategy for the model.
    """

    def __init__(
        self,
        module: Type[SynthesisModule],
        logger: logging.Logger,
        output_dir: str,
        global_config: Dict[str, Any],
        model_config: Type[AbstractModelConfig],
        reconstruction_dataloader: Optional[DataLoader] = None,
    ) -> None:
        """
        Initialize the InferenceStrategy. This is an interface class for all
        inference strategies.

        Args:
            module (Type[SynthesisModule]): The module to use for inference.
            logger (logging.Logger): The logger to use for logging.
            output_dir (str): The output directory.
            global_config (Dict[str, Any]): The global configuration dictionary.
            model_config (Type[AbstractModelConfig]): The model configuration class.
            reconstruction_dataloader (DataLoader, optional): The dataloader for the reconstruction data
        for cases where the model is an autoencoder or performs any type of reconstruction. Defaults to None.
            metric_calculator (Optional[object], optional): The metric calculator to use for calculating
        metrics during inference. Generally used only during reconstruction. Defaults to None.
        """

        self.module = module
        self.logger = logger
        self.output_dir = output_dir
        self.global_config = global_config
        self.model_config = model_config
        self.reconstruction_dataloader = reconstruction_dataloader
        self.device = module.device

    @abstractmethod
    def run_inference(self):
        """
        Run the inference strategy.
        """
        pass


class UnlabeledGenerationInferenceStrategy(InferenceStrategy):
    def run_inference(self):
        """
        Perform generation of images of the data without labels.
        """

        inference_config = self.global_config["inference_parameters"]
        n_images_to_generate = inference_config["n_images_to_generate"]
        self.logger.info(f"Generating {n_images_to_generate} images.")
        batch_size = self.global_config.get("batch_size", 1)
        n_batches, remainder = determine_n_batches(n_images_to_generate, batch_size)
        # determine how many batches are needed

        # generate the images
        for i in range(n_batches):
            n_images_batch = batch_size
            if remainder > 0 and i == n_batches - 1:
                n_images_batch = remainder
            inference_step_kwargs = {"n_images_to_generate": n_images_batch}
            generated_images = self.module.inference_step(**inference_step_kwargs)
            generated_images = prepare_images_for_saving(
                generated_images, n_dimensions=self.model_config.n_dimensions
            )
            # TODO can we make it distributed? Would be much faster
            for j, generated_image in enumerate(generated_images):
                image_path = os.path.join(
                    self.output_dir, f"generated_image_{i * batch_size + j}"
                )
                save_single_image(
                    generated_image,
                    image_path,
                    self.global_config["modality"],
                    self.model_config.n_dimensions,
                )


class LabeledGenerationInferenceStrategy(InferenceStrategy):
    def run_inference(self):
        """
        Perform generation of images of the data with labels.
        """

        inference_config = self.global_config["inference_parameters"]
        per_class_n_images_to_generate = inference_config["n_images_to_generate"]
        for class_label, n_images_to_generate in per_class_n_images_to_generate.items():
            self.logger.info(
                f"Generating {n_images_to_generate} images for class {class_label}."
            )
            # determine how many batches are needed
            batch_size = self.global_config.get("batch_size", 1)

            n_batches, remainder = determine_n_batches(n_images_to_generate, batch_size)
            # generate the images
            for i in range(n_batches):
                n_images_batch = batch_size
                if remainder > 0 and i == n_batches - 1:
                    n_images_batch = remainder
                inference_step_kwargs = {
                    "n_images_to_generate": n_images_batch,
                    "class_label": class_label,
                }
                generated_images = self.module.inference_step(**inference_step_kwargs)
                generated_images = prepare_images_for_saving(
                    generated_images, n_dimensions=self.model_config.n_dimensions
                )
                for j, generated_image in enumerate(generated_images):
                    image_path = os.path.join(
                        self.output_dir,
                        f"generated_image_{i * batch_size + j}_class_{class_label}",
                    )
                    save_single_image(
                        generated_image,
                        image_path,
                        self.global_config["modality"],
                        self.model_config.n_dimensions,
                    )


# Generally for reconstruction it is always the same, how the batch
# is structured and unpacked is on the side of the model.
class ReconstructionInferenceStrategy(InferenceStrategy):
    def run_inference(self):
        """
        Perform the reconstruction of the data without labels.
        """
        for batch_idx, batch in tqdm(
            enumerate(self.reconstruction_dataloader),
            total=len(self.reconstruction_dataloader),
            desc="Inference",
        ):
            batch = ensure_device_placement(batch, self.device)
            inference_step_kwargs = {"input_batch": batch}
            recon_images = self.module.inference_step(**inference_step_kwargs)
            recon_images = prepare_images_for_saving(
                recon_images, n_dimensions=self.model_config.n_dimensions
            )

            for i, recon_image in enumerate(recon_images):
                image_path = os.path.join(
                    self.output_dir, f"synthetic_image_{batch_idx * len(batch) + i}"
                )
                save_single_image(
                    recon_image,
                    image_path,
                    self.global_config["modality"],
                    self.model_config.n_dimensions,
                )


class InferenceStrategyFactory:
    def __init__(
        self,
        module: Type[SynthesisModule],
        logger: logging.Logger,
        output_dir: str,
        global_config: Dict[str, Any],
        model_config: Type[AbstractModelConfig],
        reconstruction_dataloader: Optional[DataLoader] = None,
    ) -> None:
        """
        Factory class to create the inference strategy. It automatically determines the strategy based on the
        configuration. It does this based on the chosen labeling paradigm in the model configuration and
        chosen model type.
        """

        self.module = module
        self.logger = logger
        self.output_dir = output_dir
        self.global_config = global_config
        self.model_config = model_config
        self.reconstruction_dataloader = reconstruction_dataloader

    def get_strategy(self) -> Type[InferenceStrategy]:
        strategy_class = self._determine_strategy_type()

        return strategy_class(
            self.module,
            self.logger,
            self.output_dir,
            self.global_config,
            self.model_config,
            self.reconstruction_dataloader,
        )

    # TODO: Think if this is the way to do it.
    def _determine_strategy_type(self) -> Type[InferenceStrategy]:
        labeling_paradigm = self.model_config.labeling_paradigm
        module_type = (
            "reconstruction"
            if self.reconstruction_dataloader is not None
            else "generation"
        )
        if module_type == "reconstruction":
            return ReconstructionInferenceStrategy
        elif labeling_paradigm == "unlabeled" and module_type == "generation":
            return UnlabeledGenerationInferenceStrategy
        elif labeling_paradigm == "labeled" and module_type == "generation":
            return LabeledGenerationInferenceStrategy
