import os
import shutil
import pickle
from warnings import warn
from logging import Logger

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchio.transforms import Compose

from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.modules.module_factory import ModuleFactory
from gandlf_synth.data.datasets_factory import DatasetFactory
from gandlf_synth.data.dataloaders_factory import DataloaderFactory
from gandlf_synth.metrics import get_metrics
from gandlf_synth.data.preprocessing import get_preprocessing_transforms
from gandlf_synth.data.augmentations import get_augmentation_transforms
from gandlf_synth.data.postprocessing import get_postprocessing_transforms
from gandlf_synth.utils.io_utils import save_single_image
from abc import ABC, abstractmethod

from typing import List, Optional, Type, Callable, Dict, Union


class AbstractInferencer(ABC):
    def __init__(self, inference_config: dict) -> None:
        self.inference_config = inference_config

    @abstractmethod
    def infer(self):
        pass


class InferenceManager:
    """Class to manage the inference of the model on the input data."""

    def __init__(
        self,
        global_config: dict,
        model_config: Type[AbstractModelConfig],
        model_dir: str,
        output_dir: str,
        device: str,
        dataframe_reconstruction: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize the InferenceManager.

        Args:
            global_config (dict): The global configuration dictionary.
            model_config (Type[AbstractModelConfig]): The model configuration class.
            model_dir (str): The directory of the run where the target model is saved.
            output_dir (str): The directory where the output files will be saved.
            device (str, optional): The device to use for inference.
            dataframe_reconstruction (Optional[pd.DataFrame], optional): The dataframe with the data
        to perform reconstruction on. This will be used only for autoencoder-style models.
        """

        self.global_config = global_config
        self.model_config = model_config
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = device
        self.dataframe_reconstruction = dataframe_reconstruction
        self._validate_inference_config()
        self.logger = self._prepare_logger()
        self.metric_calculator = self._prepare_metric_calculator()

        module_factory = ModuleFactory(
            model_config=self.model_config,
            logger=self.logger,
            device=self.device,
            model_dir=self.model_dir,
            device=self.device,
            postprocessing_transforms=self._prepare_postprocessing_transforms(),
        )
        self.module = module_factory.get_module()
        self._load_model_checkpoint()
        # ensure model in eval mode
        self.module.model.eval()
        if self.dataframe_reconstruction is not None:
            self.dataloader = self._prepare_dataloader()

    def _prepare_metric_calculator(self) -> dict:
        """
        Prepare the metric calculator for the training process.

        Returns:
            dict: The dictionary of metrics to be calculated.
        """
        return get_metrics(self.global_config["metrics"])

    def _prepare_logger(self) -> Logger:
        """
        Prepare the logger for the training process.
        """
        logger = Logger("GandlfSynthTrainingManager")
        return logger

    # TODO: should we allow the user to maybe even specify the checkpoint to load?
    def _load_model_checkpoint(self):
        """
        Resume the training process from a previous checkpoint if `resume` mode is used. This function
        establishes which model checkpoint to load and loads it.
        """

        initial_model_path = os.path.exists(
            os.path.join(self.output_dir, "model_initial.tar.gz")
        )
        latest_model_path_exists = os.path.exists(
            os.path.join(self.output_dir, "model_latest.tar.gz")
        )
        if latest_model_path_exists:
            self.logger.info("Resuming training from the latest checkpoint.")
            self.module.load_checkpoint(suffix="latest")
        elif initial_model_path:
            self.logger.info("Resuming training from the initial checkpoint.")
            self.module.load_checkpoint(suffix="initial")
        else:
            self.logger.info(
                "No model checkpoint found in the model directory, training from scratch."
            )

    def _prepare_transforms(self) -> Union[Compose, None]:
        """Prepare the transforms for the inference process in case of
        reconstruction data provided. Only preprocessing will be applied."""
        transforms_list = []
        preprocessing_operations = None
        preprocessing_config = self.global_config.get("data_preprocessing")
        if preprocessing_config:
            preprocessing_operations = preprocessing_config.get("inference")
        if preprocessing_operations:
            transforms_list.extend(
                get_preprocessing_transforms(preprocessing_operations)
            )
        if len(transforms_list) > 0:
            return Compose(transforms_list)

    def _prepare_dataloader(self) -> DataLoader:
        """
        Prepare the dataloader for the inference process if reconstruction
        data is provided.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the inference process.
        """
        transforms = self._prepare_transforms()
        dataset_factory = DatasetFactory()
        dataloader_factory = DataloaderFactory(params=self.global_config)
        dataset = dataset_factory.get_dataset(
            self.dataframe_reconstruction,
            transforms,
            labeling_paradigm=self.model_config.labeling_paradigm,
        )
        dataloader = dataloader_factory.get_inference_dataloader(dataset=dataset)
        return dataloader

    def _prepare_postprocessing_transforms(self) -> List[Callable]:
        """
        Prepare the postprocessing transforms.

        Returns:
            List[Callable]: The list of postprocessing transforms.
        """
        postprocessing_transforms = None
        postprocessing_config = self.global_config.get("data_postprocessing")
        if postprocessing_config is not None:
            postprocessing_transforms = get_postprocessing_transforms(
                postprocessing_config
            )
        return postprocessing_transforms

    # TODO: probably needs extension in the future
    def _validate_inference_config(self):
        """
        Validate the inference config.
        """
        inference_config: dict = self.global_config.get("inference_parameters")
        assert (
            "n_images_to_generate" in inference_config.keys()
        ), "The inference config for the unlabeled paradigm must contain the key 'n_images_to_generate'."
        if self.model_config.labeling_paradigm == "unlabeled":
            assert isinstance(
                inference_config["n_images_to_generate"], int
            ), "The number of images to generate must be an integer."
            assert (
                inference_config["n_images_to_generate"] > 0
            ), "The number of images to generate must be greater than 0."
        else:
            assert isinstance(
                inference_config["n_images_to_generate"], dict
            ), "The number of images to generate must be a dictionary."
            for class_label, n_images in inference_config[
                "n_images_to_generate"
            ].items():
                assert isinstance(
                    n_images, int
                ), "The number of images to generate must be an integer."
                assert (
                    n_images > 0
                ), "The number of images to generate must be greater than 0."

    def _unlabeled_inference(self, inference_config: dict):
        """
        Perform inference on the unlabeled data.

        Args:
            inference_config (dict): The inference configuration dictionary.
        """

        # get total images to generate and batch size
        n_images_to_generate = inference_config["n_images_to_generate"]
        self.logger.info(f"Generating {n_images_to_generate} images.")
        batch_size = self.global_config.get("batch_size", 1)
        # determine how many batches are needed
        n_batches = n_images_to_generate // batch_size
        remainder = n_images_to_generate % batch_size
        if remainder > 0:
            n_batches += 1
        # generate the images
        for i in range(n_batches):
            n_images_batch = batch_size
            if remainder > 0 and i == n_batches - 1:
                n_images_batch = remainder
            inference_step_kwargs = {"n_images_to_generate": n_images_batch}
            generated_images = self.module.inference_step(**inference_step_kwargs)
            # saving
            if self.model_config.n_dimensions == 2:
                generated_images = generated_images.permute(0, 2, 3, 1)
                generated_images = generated_images.numpy().astype(np.uint8)
            elif self.model_config.n_dimensions == 3:
                generated_images = generated_images.permute(0, 2, 3, 4, 1)
            # TODO can we make it distributed? Would be much faster
            for j, generated_image in enumerate(generated_images):
                image_path = os.path.join(
                    self.output_dir, f"generated_image_{i * batch_size + j}"
                )
                save_single_image(
                    generated_image,
                    image_path,
                    self.model_config.modality,
                    self.model_config.n_dimensions,
                )

    def _labeled_inference(self, inference_config: dict):
        """
        Perform inference on the labeled data.

        Args:
            inference_config (dict): The inference configuration dictionary.
        """

        n_images_to_generate = inference_config["n_images_to_generate"]
        for class_label, n_images in n_images_to_generate.items():
            self.logger.info(f"Generating {n_images_to_generate} images.")
            batch_size = self.global_config.get("batch_size", 1)
            # determine how many batches are needed
            n_batches = n_images // batch_size
            remainder = n_images % batch_size
            if remainder > 0:
                n_batches += 1
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
                # saving
                if self.model_config.n_dimensions == 2:
                    generated_images = generated_images.permute(0, 2, 3, 1)
                    generated_images = generated_images.numpy().astype(np.uint8)
                elif self.model_config.n_dimensions == 3:
                    generated_images = generated_images.permute(0, 2, 3, 4, 1)
                for j, generated_image in enumerate(generated_images):
                    image_path = os.path.join(
                        self.output_dir, f"generated_image_{i * batch_size + j}"
                    )
                    save_single_image(
                        generated_image,
                        image_path,
                        self.model_config.modality,
                        self.model_config.n_dimensions,
                    )

    def infer(self):
        """
        Perform inference on the data.
        """
        self.logger.info("Starting inference.")
        inference_config = self.global_config["inference_parameters"]
        if self.model_config.labeling_paradigm == "unlabeled":
            self._unlabeled_inference(inference_config)
        else:
            self._labeled_inference(inference_config)
        self.logger.info("Inference finished.")
