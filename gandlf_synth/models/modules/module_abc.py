import os
import io
import tarfile
import hashlib
from logging import Logger
from abc import ABC, abstractmethod, ABCMeta

import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from GANDLF.utils.generic import get_git_hash, get_unique_timestamp

from gandlf_synth.version import __version__
from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.utils.compute import ensure_device_placement

from typing import Dict, Union, Optional, Type, List, Callable


class SynthesisModule(pl.LightningModule, metaclass=ABCMeta):
    """Abstract class for a synthesis module. It wraps the model architecture
    and logic assocaited with different steps (i.e. forward pass, training_step etc.).
    Uses Pytorch Lightning as the base class, with extra functionality added on top.
    """

    def __init__(
        self,
        model_config: Type[AbstractModelConfig],
        model_dir: str,
        metric_calculator: Optional[Dict[str, Callable]] = None,
        postprocessing_transforms: Optional[List[Callable]] = None,
    ) -> None:
        """Initialize the synthesis module.

        Args:
            params (dict): Dictionary of parameters.
            logger (Logger): Logger for logging the values.
            model_dir (str) : Model and results output directory.
            metric_calculator (Dict[str,Callable],optional): Metric calculator object.
            postprocessing_transforms (List[Callable], optional): Postprocessing transformations to apply.
        """

        super().__init__()

        # This is my idea for now, we can change it later.
        self.model_config = model_config
        self.model_dir = model_dir
        self.metric_calculator = metric_calculator
        self.postprocessing_transforms = postprocessing_transforms
        self.model = self._initialize_model()
        self.losses = self._initialize_losses()
        # Ensure the objects are placed on the device.
        self.model = ensure_device_placement(self.model, self.device)
        self.losses = ensure_device_placement(self.losses, self.device)

    @abstractmethod
    def _initialize_model(self) -> ModelBase:
        """
        Initialize the model for the synthesis module.

        Returns:
            model (ModelBase): Model for the synthesis module.
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
        if isinstance(self.optimizers, optim.Optimizer):
            optimizers_state_dict = self.optimizers.state_dict()
        else:
            optimizers_state_dict = {
                key: value.state_dict() for key, value in self.optimizers.items()
            }
        schedulers_state_dict = None

        if self.schedulers is not None:
            if isinstance(self.schedulers, optim.lr_scheduler._LRScheduler):
                schedulers_state_dict = self.schedulers.state_dict()
            else:
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
        torch_object_filename = "model-" + suffix.strip("_") + ".pt"
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

        def _determine_checkpoint_to_load(output_dir: str) -> Union[str, None]:
            """
            Based on the present checkpoints, determine which checkpoint to load.
            If a custom suffix is provided, it will be used to load the checkpoint.
            Args:
                output_dir (str): The output directory for the model.
            Returns:
                suffix (Union[str,None]): The suffix of the checkpoint to load if any.
            """
            best_model_path_exists = os.path.exists(
                os.path.join(output_dir, "model-best.tar.gz")
            )
            latest_model_path_exists = os.path.exists(
                os.path.join(output_dir, "model-latest.tar.gz")
            )
            initial_model_path_exists = os.path.exists(
                os.path.join(output_dir, "model-initial.tar.gz")
            )
            suffix = None
            if best_model_path_exists:
                suffix = "best"
            elif latest_model_path_exists:
                suffix = "latest"
            elif initial_model_path_exists:
                suffix = "initial"
            return suffix

        if suffix is None:
            suffix = _determine_checkpoint_to_load(self.model_dir)
        if suffix is None:
            self.logger.info(
                "No checkpoint found in the model directory. Skipping loading the model."
            )
            return
        MAP_LOCATION_DICT = {
            "state_dict": self.device,
            "optimizers_state_dict": "cpu",
            "schedulers_state_dict": "cpu",
            "timestamp": "cpu",
            "timestamp_hash": "cpu",
            "version": "cpu",
            "git_hash": "cpu",
        }
        tar_file_path = os.path.join(self.model_dir, "model-" + suffix + ".tar.gz")
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
        if isinstance(self.optimizers, optim.Optimizer):
            self.optimizers.load_state_dict(optimizers_state_dict)
        else:
            for name, optimizer in self.optimizers.items():
                assert (
                    name in optimizers_state_dict.keys()
                ), f"Optimizer {name} not found in the checkpoint!"
                optimizer.load_state_dict(optimizers_state_dict[name])
        if self.schedulers is not None:
            schedulers_state_dict = metadata_dict["schedulers_state_dict"]
            if schedulers_state_dict is not None:
                if isinstance(self.schedulers, optim.lr_scheduler._LRScheduler):
                    self.schedulers.load_state_dict(schedulers_state_dict)
                else:
                    for name, scheduler in self.schedulers.items():
                        assert (
                            name in schedulers_state_dict.keys()
                        ), f"Scheduler {name} not found in the checkpoint!"
                        scheduler.load_state_dict(schedulers_state_dict[name])
        self.logger.info(f"Model loaded from {tar_file_path}")
        self.logger.info(f"GANDLF-Synth version: {metadata_dict['version']}")
        self.logger.info(f"Git hash: {git_hash}")
        self.logger.info(f"Timestamp: {timestamp}")
        self.logger.info(f"Timestamp hash: {timestamp_hash}")

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
