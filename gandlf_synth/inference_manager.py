import os
import pandas as pd

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.modules.module_factory import ModuleFactory
from gandlf_synth.data.datasets_factory import InferenceDatasetFactory
from gandlf_synth.utils.managers_utils import (
    prepare_logger,
    prepare_postprocessing_transforms,
)
from typing import Optional, Type


class CustomPredictionImageSaver(pl.callbacks.BasePredictionWriter):
    # TODO here it needs to save images
    # ensure that it properly saves the images to inference output dir
    # ensure that it save images with correct class label in case of labeled inference
    pass


class InferenceManager:
    LOGGER_NAME = "InferenceManager"

    """Class to manage the inference of the model on the input data."""

    def __init__(
        self,
        global_config: dict,
        model_config: Type[AbstractModelConfig],
        model_dir: str,
        output_dir: str,
        dataframe_reconstruction: Optional[pd.DataFrame] = None,
        custom_checkpoint_path: Optional[str] = None,
    ) -> None:
        """Initialize the InferenceManager.

        Args:
            global_config (dict): The global configuration dictionary.
            model_config (Type[AbstractModelConfig]): The model configuration class.
            model_dir (str): The directory of the run where the target model is saved.
            output_dir (str): The top directory where the output files will be saved. The
            inference results will be saved in a subdirectory of this directory, with name equal
            to the model_dir basename.
            device (str, optional): The device to use for inference.
            dataframe_reconstruction (Optional[pd.DataFrame], optional): The dataframe with the data
        to perform reconstruction on. This will be used only for autoencoder-style models.
            custom_checkpoint_path (Optional[str], optional): The custom suffix for the checkpoint,
        mostly used when the model is to be loaded from specific epoch checkpoint.
        """

        self.global_config = global_config
        self.model_config = model_config
        self.model_dir = model_dir
        self.output_dir = self._prepare_output_directory(output_dir, model_dir)
        self.dataframe_reconstruction = dataframe_reconstruction
        self.logger = prepare_logger(self.LOGGER_NAME, self.output_dir)

        module_factory = ModuleFactory(
            model_config=self.model_config,
            model_dir=self.model_dir,
            postprocessing_transforms=prepare_postprocessing_transforms(
                global_config=self.global_config
            ),
        )
        dataset_factory = InferenceDatasetFactory(
            global_config=self.global_config,
            model_config=self.model_config,
            dataframe_reconstruction=self.dataframe_reconstruction,
        )
        inference_dataset = dataset_factory.get_inference_dataset()
        self.inference_dataloader = self._prepare_inference_dataloader(
            inference_dataset
        )
        self.module = module_factory.get_module()
        inference_logger = pl.loggers.CSVLogger(
            self.output_dir, name="Inference_logs", flush_logs_every_n_steps=10
        )
        self.trainer = pl.Trainer(logger=inference_logger)
        self.custom_checkpoint_path = custom_checkpoint_path
        self.checkpoint_path = self._determine_checkpoint_to_load()

    @staticmethod
    def _prepare_output_directory(output_dir: str, model_dir: str) -> str:
        """
        Prepare the output directory to save inference results. If the directory
        does not exist, it will be created. If it exists, new directory will be
        created with a new index.
        """

        def _prepare_out_dir_path(output_dir: str, model_dir: str) -> str:
            """
            Prepare the save path for the inference results by merging the output
            directory and the model directory name.

            Args:
                output_dir (str): The output directory.
                model_dir (str): The model directory name.

            Returns:
                str: The save path for the inference results.
            """
            return os.path.join(
                output_dir, os.path.basename(model_dir) + "_inference_output"
            )

        model_inference_output_path = _prepare_out_dir_path(output_dir, model_dir)

        if not os.path.exists(model_inference_output_path):
            os.makedirs(model_inference_output_path)
            return model_inference_output_path

        index = 1
        while os.path.exists(f"{model_inference_output_path}_{index}"):
            index += 1
        model_inference_output_path = f"{model_inference_output_path}_{index}"
        os.makedirs(model_inference_output_path)
        return model_inference_output_path

    def _determine_checkpoint_to_load(self) -> str:
        """
        Determine the checkpoint to load for the inference process. If the custom
        checkpoint path is provided, it will be used. Otherwise, the best checkpoint
        will be loaded.
        """
        if self.custom_checkpoint_path is not None:
            return self.custom_checkpoint_path
        best_checkpoint_path = os.path.join(self.model_dir, "checkpoints", "best.ckpt")
        if os.path.exists(best_checkpoint_path):
            return best_checkpoint_path
        last_checkpoint_path = os.path.join(self.model_dir, "last.ckpt")
        if os.path.exists(last_checkpoint_path):
            return last_checkpoint_path

    def _prepare_inference_dataloader(self, dataset) -> DataLoader:
        """
        Prepare the dataloader for the inference process if reconstruction
        data is provided.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the inference process.
        """
        inference_dataloader_config = self.global_config["dataloader_config"][
            "inference"
        ]
        inference_parameters = self.global_config.get("inference_parameters")
        if inference_parameters is not None:
            inference_batch_size = inference_parameters["batch_size"]
        else:
            inference_batch_size = self.global_config["batch_size"]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=inference_batch_size,
            **inference_dataloader_config,
        )
        return dataloader

    def run_inference(self):
        """
        Perform inference on the data.
        """
        self.trainer.predict(
            self.module,
            dataloaders=self.inference_dataloader,
            ckpt_path=self.checkpoint_path,
        )
