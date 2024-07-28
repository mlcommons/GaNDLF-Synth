import os
import shutil
import pickle
from tqdm import tqdm
from warnings import warn

import torch
import pandas as pd

from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.modules.module_factory import ModuleFactory
from gandlf_synth.data.datasets_factory import DatasetFactory
from gandlf_synth.data.dataloaders_factory import DataloaderFactory
from gandlf_synth.utils.managers_utils import (
    prepare_logger,
    prepare_postprocessing_transforms,
    prepare_transforms,
    assert_input_correctness,
)
from gandlf_synth.utils.compute import ensure_device_placement
from gandlf_synth.metrics import get_metrics

from typing import Optional, Type


class TrainingManager:
    LOGGER_NAME = "training_manager"
    """
    A class to manage the training process of a model. This class ties all the necessary
    components together to train a model.
    """

    def __init__(
        self,
        train_dataframe: pd.DataFrame,
        output_dir: str,
        global_config: dict,
        model_config: Type[AbstractModelConfig],
        resume: bool,
        reset: bool,
        device: str,
        val_dataframe: Optional[pd.DataFrame] = None,
        test_dataframe: Optional[pd.DataFrame] = None,
        val_ratio: Optional[float] = 0,
        test_ratio: Optional[float] = 0,
        custom_checkpoint_suffix: Optional[str] = None,
    ):
        """
        Initialize the TrainingManager.

        Args:
            train_dataframe (pd.DataFrame): The training dataframe.
            output_dir (str): The main output directory.
            global_config (dict): The global configuration dictionary.
            model_config (Type[AbstractModelConfig]): The model configuration object.
            resume (bool): Whether the previous run will be resumed or not.
            reset (bool): Whether the previous run will be reset or not.
            device (str): The device to perform computations on.
            val_dataframe (pd.DataFrame, optional): The validation dataframe. Defaults to None.
            test_dataframe (pd.DataFrame, optional): The test dataframe. Defaults to None.
            val_ratio (float, optional): The percentage of data to be used for validation,
        extracted from the training dataframe. This parameter will be used if val_dataframe is None.
        If test_ratio is also specified, testing data will be extracted first, and then the
        remaining data will be split into training and validation data. Defaults to 0.
            test_ratio (float, optional): The percentage of data to be used for testing,
        extracted from the training dataframe. This parameter will be used if test_dataframe is None. Defaults to 0.
            custom_checkpoint_suffix (str, optional): The custom suffix to resume training from a specific checkpoint.
        Used only if resume is True. Defaults to None.
        """

        self.train_dataframe = train_dataframe
        self.val_dataframe = val_dataframe
        self.test_dataframe = test_dataframe
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.output_dir = output_dir
        self.global_config = global_config
        self.model_config = model_config
        self.resume = resume
        self.reset = reset
        self.device = device

        self.logger = prepare_logger(self.LOGGER_NAME)
        self._prepare_output_dir()
        self._load_or_save_parameters()
        self._assert_parameter_correctness()
        self._warn_user()

        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = self._prepare_dataloaders()
        metric_calculator = get_metrics(global_config["metrics"]) if "metrics" in global_config else None
        # TODO move it to the main_run function, as well as logger initialization
        module_factory = ModuleFactory(
            model_config=self.model_config,
            logger=self.logger,
            model_dir=self.output_dir,
            metric_calculator=metric_calculator,
            device=self.device,
            postprocessing_transforms=prepare_postprocessing_transforms(
                global_config=self.global_config
            ),
        )
        self.module = module_factory.get_module()
        if self.resume:
            self.module.load_checkpoint(custom_checkpoint_suffix)

    def _warn_user(self):
        """
        Warn the user about the validation and testing configuration.
        """
        if self.val_dataframe is None and self.val_ratio == 0:
            warn(
                "Validation data is not provided and the validation ratio is set to 0. "
                "The model will not be validated during the training process.",
                UserWarning,
            )
        if self.test_dataframe is None and self.test_ratio == 0:
            warn(
                "Test data is not provided and the test ratio is set to 0. "
                "The model will not be tested after the training process.",
                UserWarning,
            )
        if self.val_dataframe is not None and self.val_ratio != 0:
            warn(
                "Validation data is provided and the validation ratio is set to a non-zero value. "
                "The validation data provided will be used for validation, and the validation ratio will be ignored.",
                UserWarning,
            )
        if self.test_dataframe is not None and self.test_ratio != 0:
            warn(
                "Test data is provided and the test ratio is set to a non-zero value. "
                "The test data provided will be used for testing, and the test ratio will be ignored.",
                UserWarning,
            )
        if self.val_dataframe is None and self.val_ratio != 0:
            warn(
                "Validation data is not provided, and the validation ratio is set to a non-zero value. "
                "Validation data will be extracted from the training data."
                "IMPORTANT: samples from the training data will be RANDOMLY selected REGARDLESS of the subjects they come from."
                "If you want to avoid samples from the same subject to be split between training and validation, provide a validation dataframe.",
                UserWarning,
            )
        if self.test_dataframe is None and self.test_ratio != 0:
            warn(
                "Test data is not provided, and the test ratio is set to a non-zero value. "
                "Test data will be extracted from the training data."
                "IMPORTANT: samples from the training data will be RANDOMLY selected REGARDLESS of the subjects they come from."
                "If you want to avoid samples from the same subject to be split between training and testing, provide a test dataframe.",
                UserWarning,
            )

    def _assert_parameter_correctness(self):
        """
        Assert the correctness of the parameters.
        """
        assert (
            self.val_ratio >= 0 and self.val_ratio <= 1
        ), "Validation ratio must be between 0 and 1"
        assert (
            self.test_ratio >= 0 and self.test_ratio <= 1
        ), "Test ratio must be between 0 and 1"
        assert (
            self.val_ratio + self.test_ratio <= 1
        ), "Validation and test ratios must sum up to less than or equal to 1"
        assert self.device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'"
        if self.device == "cuda":
            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Please use 'cpu' or install CUDA."
        if self.reset and self.resume:
            warn(
                "Both reset and resume are set to True. Resume will take precedence.",
                UserWarning,
            )
            self.reset = False

    def _load_or_save_parameters(self):
        """
        Load or save the parameters for the training process.
        """
        parameters_pickle_path = os.path.join(self.output_dir, "parameters.pkl")
        pickle_file_exists = os.path.exists(parameters_pickle_path)

        if self.resume and pickle_file_exists:
            self.logger.info("Resuming training from previous run, loading parameters.")
            with open(parameters_pickle_path, "rb") as pickle_file:
                loaded_parameters = pickle.load(pickle_file)
            self.global_config = loaded_parameters["global_config"]
            self.model_config = loaded_parameters["model_config"]

        else:
            self.logger.info("Saving parameters for the current run.")
            with open(parameters_pickle_path, "wb") as pickle_file:
                pickle.dump(
                    {
                        "global_config": self.global_config,
                        "model_config": self.model_config,
                    },
                    pickle_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def _prepare_output_dir(self):
        """
        Prepare the output directory for the training process.
        """
        if self.reset and os.path.exists(self.output_dir):
            self.logger.info(
                f"Reset flag chosen, erasing {self.output_dir} and training from scratch."
            )
            shutil.rmtree(self.output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def _extract_random_data_from_dataframe(dataframe: pd.DataFrame, ratio: float):
        """
        Extracts random data indices from the dataframe based on the ratio.
        Chosen indices are removed from the original dataframe in place.
        Args:
            dataframe (pd.DataFrame): The dataframe to extract data from.
            ratio (float): The ratio of data to be extracted.

        Returns:
            pd.DataFrame: The extracted data.
        """
        num_samples_to_extract = int(len(dataframe) * ratio)
        random_rows = dataframe.sample(num_samples_to_extract, replace=False)
        dataframe.drop(random_rows.index, inplace=True)
        new_dataframe = pd.DataFrame(random_rows, columns=dataframe.columns)

        return new_dataframe

    def _prepare_dataloaders(self) -> tuple:
        """
        Prepare the dataloaders for the training, validation, and testing datasets.
        """
        dataset_factory = DatasetFactory()
        dataloader_factory = DataloaderFactory(params=self.global_config)
        preprocessing_config = self.global_config.get("data_preprocessing")
        augmentations_config = self.global_config.get("data_augmentations")

        # Extract validation and test data if not provided and ratios are specified
        if self.test_dataframe is None and self.test_ratio != 0:
            self.test_dataframe = self._extract_random_data_from_dataframe(
                self.train_dataframe, self.test_ratio
            )
        if self.val_dataframe is None and self.val_ratio != 0:
            self.val_dataframe = self._extract_random_data_from_dataframe(
                self.train_dataframe, self.val_ratio
            )
        train_transforms = prepare_transforms(
            preprocessing_config,
            augmentations_config,
            "train",
            self.model_config.tensor_shape,
        )
        train_dataset = dataset_factory.get_dataset(
            self.train_dataframe, train_transforms, self.model_config.labeling_paradigm
        )
        train_dataloader = dataloader_factory.get_training_dataloader(train_dataset)
        # Here we need to consider cases where user did not specify val or test dataframes
        val_dataloader = None
        test_dataloader = None
        if self.val_dataframe is not None:
            val_transforms = prepare_transforms(
                preprocessing_config,
                augmentations_config,
                "val",
                self.model_config.tensor_shape,
            )
            val_dataset = dataset_factory.get_dataset(
                self.val_dataframe, val_transforms, self.model_config.labeling_paradigm
            )
            val_dataloader = dataloader_factory.get_validation_dataloader(val_dataset)
        if self.test_dataframe is not None:
            test_transforms = prepare_transforms(
                preprocessing_config,
                augmentations_config,
                "test",
                self.model_config.tensor_shape,
            )
            test_dataset = dataset_factory.get_dataset(
                self.test_dataframe,
                test_transforms,
                self.model_config.labeling_paradigm,
            )
            test_dataloader = dataloader_factory.get_testing_dataloader(test_dataset)

        return train_dataloader, val_dataloader, test_dataloader

    def run_training(self):
        """
        Train the model.
        """
        # CAUTION - keep careful when dealing with multiple batches and split images (slices)
        self.module.save_checkpoint(suffix="_start")

        for epoch in range(self.global_config["num_epochs"]):
            self.module._on_train_epoch_start(epoch)

            for batch_idx, batch in tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Training epoch {epoch}/{self.global_config['num_epochs']}",
            ):
                assert_input_correctness(
                    configured_input_shape=self.model_config.tensor_shape,
                    configured_n_channels=self.model_config.n_channels,
                    batch_idx=batch_idx,
                    batch=batch,
                )
                batch = ensure_device_placement(batch, self.device)
                self.module.training_step(batch, batch_idx)
            self.module._on_train_epoch_end(epoch)
            if self.val_dataloader is not None:
                self.module._on_validation_epoch_start(epoch)
                for batch_idx, batch in tqdm(
                    enumerate(self.val_dataloader),
                    total=len(self.val_dataloader),
                    desc=f"Validation epoch {epoch}/{self.global_config['num_epochs']}",
                ):
                    assert_input_correctness(
                        configured_input_shape=self.model_config.tensor_shape,
                        configured_n_channels=self.model_config.n_channels,
                        batch_idx=batch_idx,
                        batch=batch,
                    )
                    batch = ensure_device_placement(batch, self.device)
                    self.module.validation_step(batch, batch_idx)
                self.module._on_validation_epoch_end(epoch)
            if self.global_config["save_model_every_n_epochs"] != -1 and (
                epoch % self.global_config["save_model_every_n_epochs"] == 0
            ):
                self.module.save_checkpoint(suffix=f"epoch-{epoch}")
            self.module.save_checkpoint(suffix=f"latest")
        if self.test_dataloader is not None:
            self.module._on_test_start()
            for batch_idx, batch in tqdm(
                enumerate(self.test_dataloader),
                total=len(self.test_dataloader),
                desc="Testing",
            ):
                assert_input_correctness(
                    configured_input_shape=self.model_config.tensor_shape,
                    configured_n_channels=self.model_config.n_channels,
                    batch_idx=batch_idx,
                    batch=batch,
                )
                batch = ensure_device_placement(batch, self.device)
                self.module.test_step(batch, batch_idx)
            self.module._on_test_end()
