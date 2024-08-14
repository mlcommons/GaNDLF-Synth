import os
import pandas as pd

from torch.utils.data import DataLoader

from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.modules.module_factory import ModuleFactory
from gandlf_synth.data.datasets_factory import DatasetFactory
from gandlf_synth.data.dataloaders_factory import DataloaderFactory
from gandlf_synth.utils.managers_utils import (
    prepare_logger,
    prepare_postprocessing_transforms,
    prepare_transforms,
)
from gandlf_synth.utils.inference_strategies import InferenceStrategyFactory
from typing import Optional, Type


class InferenceManager:
    LOGGER_NAME = "InferenceManager"

    """Class to manage the inference of the model on the input data."""

    def __init__(
        self,
        global_config: dict,
        model_config: Type[AbstractModelConfig],
        model_dir: str,
        output_dir: str,
        device: str,
        dataframe_reconstruction: Optional[pd.DataFrame] = None,
        custom_checkpoint_suffix: Optional[str] = None,
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
            custom_checkpoint_suffix (Optional[str], optional): The custom suffix for the checkpoint,
        mostly used when the model is to be loaded from specific epoch checkpoint.
        """

        self.global_config = global_config
        self.model_config = model_config
        self.model_dir = model_dir
        self.output_dir = self._prepare_output_directory(output_dir, model_dir)
        self.device = device
        self.dataframe_reconstruction = dataframe_reconstruction
        self.logger = prepare_logger(self.LOGGER_NAME, self.output_dir)

        module_factory = ModuleFactory(
            model_config=self.model_config,
            logger=self.logger,
            device=self.device,
            model_dir=self.model_dir,
            postprocessing_transforms=prepare_postprocessing_transforms(
                global_config=self.global_config
            ),
        )
        self.module = module_factory.get_module()
        self.module.load_checkpoint(custom_checkpoint_suffix)
        # ensure model in eval mode
        self.module.model.eval()
        reconstruction_dataloader = None
        if self.dataframe_reconstruction is not None:
            reconstruction_dataloader = self._prepare_inference_dataloader()

        inference_strategy_factory = InferenceStrategyFactory(
            module=self.module,
            logger=self.logger,
            output_dir=self.output_dir,
            model_config=self.model_config,
            global_config=self.global_config,
            reconstruction_dataloader=reconstruction_dataloader,
        )
        self.inference_strategy = inference_strategy_factory.get_strategy()

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

    def _prepare_inference_dataloader(self) -> DataLoader:
        """
        Prepare the dataloader for the inference process if reconstruction
        data is provided.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the inference process.
        """
        transforms = prepare_transforms(
            augmentations_config=self.global_config.get("data_augmentations"),
            preprocessing_config=self.global_config.get("data_preprocessing"),
            mode="inference",
            input_shape=self.model_config.tensor_shape,
        )
        dataset_factory = DatasetFactory()
        dataloader_factory = DataloaderFactory(params=self.global_config)
        dataset = dataset_factory.get_dataset(
            self.dataframe_reconstruction,
            transforms,
            labeling_paradigm=self.model_config.labeling_paradigm,
        )
        dataloader = dataloader_factory.get_inference_dataloader(dataset=dataset)
        return dataloader

    def run_inference(self):
        """
        Perform inference on the data.
        """
        self.logger.info("Starting inference.")
        self.inference_strategy.run_inference()
        self.logger.info("Inference finished.")
