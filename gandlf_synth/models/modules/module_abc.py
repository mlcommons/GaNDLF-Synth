import os
import io
import tarfile
import hashlib
from logging import Logger
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import optim

from GANDLF.utils.generic import get_git_hash, get_unique_timestamp

from gandlf_synth.version import __version__
from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.architectures.base_model import ModelBase

from typing import Dict, Union, Optional, Type, List, Callable


class SynthesisModule(ABC):
    """Abstract class for a synthesis module. It wraps the model architecture
    and logic assocaited with different steps (i.e. forward pass, training_step etc.).
    Greatly inspired by PyTorch Lightning.
    """

    def __init__(
        self,
        model_config: Type[AbstractModelConfig],
        logger: Logger,
        model_dir: str,
        metric_calculator: Optional[dict] = None,
        postprocessing_transforms: Optional[List[Callable]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the synthesis module.

        Args:
            params (dict): Dictionary of parameters.
            logger (Logger): Logger for logging the values.
            model_dir (str) : Model and results output directory.
            metric_calculator (object,optional): Metric calculator object.
            postprocessing_transforms (List[Callable], optional): Postprocessing transformations to apply.
            device (str, optional): Device to perform computations on. Defaults to "cpu".
        """

        super().__init__()
        # This is my idea for now, we can change it later.
        self.model_config = model_config
        self.logger = logger
        self.model_dir = model_dir
        self.metric_calculator = metric_calculator
        self.postprocessing_transforms = postprocessing_transforms
        self.device = torch.device(device)
        self.model = self._initialize_model()
        self.optimizers = self._initialize_optimizers()
        self.losses = self._initialize_losses()
        self.schedulers = self._initialize_schedulers()
        # Ensure the objects are placed on the device.
        self.model = self._ensure_device_placement(self.model)
        self.losses = self._ensure_device_placement(self.losses)

    @abstractmethod
    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        """
        Training step for the synthesis module.

        Args:
            batch: A batch of data.
            batch_idx (int): Index of the batch.
        Returns:
            loss (torch.Tensor): Loss value.
        """

        pass

    @abstractmethod
    @torch.no_grad
    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the synthesis module.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.
        Returns:
            loss (torch.Tensor): Loss value.
        """

        pass

    @abstractmethod
    def test_step(self, **kwargs) -> torch.Tensor:
        """
        Inference step for the synthesis module.

        Args:
            kwargs: Key-word arguments.
        Returns:
            output: Model output.
        """
        pass

    @abstractmethod
    def inference_step(self, **kwargs) -> torch.Tensor:
        """
        Inference step for the synthesis module.

        Args:
            kwargs: Key-word arguments.
        Returns:
            output: Model output.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the synthesis module.

        Args:
            x: Input data.
        Returns:
            output: Model output.
        """
        # Not sure if the construction is correct, probably in some
        # cases this class will accept multiple inputs.

        pass

    @abstractmethod
    def _initialize_model(self) -> ModelBase:
        """
        Initialize the model for the synthesis module.

        Returns:
            model (ModelBase): Model for the synthesis module.
        """
        pass

    @abstractmethod
    def _initialize_optimizers(
        self,
    ) -> Union[optim.Optimizer, Dict[str, optim.Optimizer]]:
        """
        Initialize the optimizer (or optimizers) for the synthesis module.
        Multiple optimizers can be defined for different parts of the model
        (e.g. generator and discriminator). Those will be initialized from the
        self.params dictionary probably.

        Returns:
            optimizer (torch.optim.Optimizer or dict): Optimizer(s) for the model.

        """
        pass

    @abstractmethod
    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        """
        Initialize the loss function (or functions) for the synthesis module.
        Multiple losses can be defined for different parts of the model
        (e.g. generator and discriminator). Those will be initialized from the
        self.params dictionary probably.

        Returns:
            loss (torch.nn.Module or dict): Loss function(s) for the model.
        """
        pass

    @abstractmethod
    def _initialize_schedulers(
        self,
    ) -> Union[
        optim.lr_scheduler._LRScheduler, Dict[str, optim.lr_scheduler._LRScheduler]
    ]:
        """
        Initialize the learning rate scheduler (or schedulers) for the synthesis module.
        Multiple schedulers can be defined for different parts of the model
        (e.g. generator and discriminator). Those will be initialized from the
        self.params dictionary probably.

        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler or dict): Scheduler(s) for
        the model.
        """
        pass

    def save_checkpoint(self, suffix: str) -> None:
        """
        Save the model checkpoint, optimizer state, scheduler state and metadata into specified run directory.
        Pytorch-serialized object is compressed into tar.gz archive.

        Args:
            suffix (str) : Suffix to be added to the basic archive name,
        used mostly for versioning. This suffix SHOULD NOT contain file
        extensions.

        """

        state_dict = self.model.state_dict()
        optimizers_state_dict = {
            key: value.state_dict() for key, value in self.optimizers.items()
        }
        schedulers_state_dict = None
        if self.schedulers is not None:
            schedulers_state_dict = {
                key: value.state_dict() for key, value in self.schedulers.items()
            }
        timestamp = get_unique_timestamp()
        timestamp_hash = hashlib.sha256(str(timestamp).encode("utf-8")).hexdigest()

        metadata_dict = {
            "version": __version__,
            "git_hash": get_git_hash(),
            "timestamp": timestamp,
            "timestamp_hash": timestamp_hash,
            "state_dict": state_dict,
            "optimizers_state_dict": optimizers_state_dict,
            "schedulers_state_dict": schedulers_state_dict,
        }
        state_dict_io = io.BytesIO()
        torch_object_filename = "model_" + suffix.strip("_") + ".pt"
        tarfile_object_filename = torch_object_filename.split(".")[0] + ".tar.gz"
        tarfile_object_path = os.path.join(self.model_dir, tarfile_object_filename)
        torch.save(metadata_dict, state_dict_io)
        state_dict_io.seek(0)
        with tarfile.open(tarfile_object_path, "w:gz") as archive:
            tarinfo = tarfile.TarInfo(torch_object_filename)
            tarinfo.size = len(state_dict_io.getbuffer())
            archive.addfile(tarinfo, state_dict_io)

    # TODO think on loading it, how to handle filename
    def load_checkpoint(self, suffix: Optional[str] = None) -> None:
        """
        Load the model checkpoint, optimizer states, scheduler states and metadata from specified run directory.
        If specieifd, add suffix to the filename to load specific ones.

        Args:
            model_dir (str) : Directory with run files stored.
            suffix (Optional[str]) : Optional suffix to be added to the filename,
        used mostly for versioning.

        """

        MAP_LOCATION_DICT = {
            "state_dict": self.device,
            "optimizers_state_dict": "cpu",
            "schedulers_state_dict": "cpu",
            "timestamp": "cpu",
            "timestamp_hash": "cpu",
            "version": "cpu",
            "git_hash": "cpu",
        }

        tar_file_path = os.path.join(self.model_dir, "model_" + suffix + ".tar.gz")
        torch_object_path = os.path.basename(tar_file_path).split(".")[0] + ".pt"
        with tarfile.open(tar_file_path, "r:gz") as archive:
            metadata_dict = torch.load(
                archive.extractfile(torch_object_path), map_location=MAP_LOCATION_DICT
            )
        timestamp = metadata_dict["timestamp"]
        timestamp_hash = metadata_dict["timestamp_hash"]
        git_hash = metadata_dict["git_hash"]
        optimizers_state_dict = metadata_dict["optimizers_state_dict"]
        self.model.load_state_dict(state_dict=metadata_dict["state_dict"])
        for name, optimizer in self.optimizers.items():
            assert (
                name in optimizers_state_dict.keys()
            ), f"Optimizer {name} not found in the checkpoint!"
            optimizer.load_state_dict(optimizers_state_dict[name])
        if self.schedulers is not None:
            schedulers_state_dict = metadata_dict["schedulers_state_dict"]
            if schedulers_state_dict is not None:
                for name, scheduler in self.schedulers.items():
                    assert (
                        name in schedulers_state_dict.keys()
                    ), f"Scheduler {name} not found in the checkpoint!"
                    scheduler.load_state_dict(schedulers_state_dict[name])
        self.logger.log(10, f"Model loaded from {tar_file_path}")
        self.logger.log(10, f"GANDLF-Synth version: {metadata_dict['version']}")
        self.logger.log(10, f"Git hash: {git_hash}")
        self.logger.log(10, f"Timestamp: {timestamp}")
        self.logger.log(10, f"Timestamp hash: {timestamp_hash}")

    def _apply_postprocessing(self, data_to_transform: torch.Tensor) -> torch.Tensor:
        """
        Applies postprocessing transformations to the data.

        Args:
            data_to_transform (torch.Tensor): Data to transform.

        Returns:
            transformed_data (torch.Tensor): Transformed data.
        """
        for self.data_transform in self.postprocessing_transforms:
            data_to_transform = self.data_transform(data_to_transform)
        return data_to_transform

    def _log(self, value_name: str, value_to_log: float) -> None:
        """
        Log the value to the logger.

        Args:
            value_name: Name of the value to log.
            value_to_log: Value to log.
        """
        # TODO : We need to think on that, I was wondering if we can use the main program logger
        # that we used in GaNDLF for logging the values. Maybe we should also wait for Sylwia's
        # port of new logging in main GaDLF. Anyway the logging should be done in the same way
        # for all the modules.
        self.logger.log(10, f"{value_name}: {value_to_log}")

    def _log_dict(self, dict_to_log: Dict[str, float]) -> None:
        """
        Log the dictionary of values to the logger.

        Args:
            dict_to_log: Dictionary of values to log.
        """
        for key, value in dict_to_log.items():
            self._log(10, f"{key}: {value}")

    def _ensure_device_placement(self, data: object) -> object:
        """
        Ensure the data is placed on the device.

        Args:
            data: Data to place on the device.
        Returns:
            data: Data placed on the device.
        """
        if isinstance(data, torch.Tensor) or isinstance(data, nn.Module):
            return data.to(self.device)
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = value.to(self.device)
            return data

    def _on_train_epoch_start(self, epoch: int) -> None:
        """
        Function to be called at the start of the epoch.

        Args:
            epoch (int): Current epoch.
        """
        pass

    def _on_validation_epoch_start(self, epoch: int) -> None:
        """
        Function to be called at the start of the validation.

        Args:
            epoch (int): Current epoch.
        """
        pass

    def _on_test_start(self) -> None:
        """
        Function to be called at the start of the test.

        """
        pass

    def _on_train_epoch_end(self, epoch: int) -> None:
        """
        Function to be called at the end of the epoch.

        Args:
            epoch (int): Current epoch.
        """
        pass

    def _on_validation_epoch_end(self, epoch: int) -> None:
        """
        Function to be called at the end of the validation.

        Args:
            epoch (int): Current epoch.
        """
        pass

    def _on_test_end(self) -> None:
        """
        Function to be called at the end of the test.
        """
        pass
