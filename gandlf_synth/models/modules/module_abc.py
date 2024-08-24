import os
from logging import Logger
from abc import abstractmethod, ABCMeta

import torch
from torch import nn
import lightning.pytorch as pl

from gandlf_synth.version import __version__
from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.architectures.base_model import ModelBase
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

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Union[None, torch.optim.lr_scheduler._LRScheduler]:
        """
        Get the scheduler for the optimizer if it is defined.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to get the scheduler for.

        Returns:
            scheduler (Union[None, torch.optim.lr_scheduler._LRScheduler]): Scheduler for the optimizer.
        """
        return None

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load the module checkpoint from specified run directory.
        If specified, loads directly from the checkpoint path, otherwise
        tries to load the latest checkpoint from the model directory.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file.

        """

        def _determine_checkpoint_to_load(output_dir: str) -> Union[str, None]:
            """
            Based on the present checkpoints, determine which checkpoint to load.
            If a custom suffix is provided, it will be used to load the checkpoint.
            Args:
                output_dir (str): The output directory for the model.
            Returns:
                checkpoint_path (str): Path to the checkpoint to load.
            """
            last_checkpoint_path = os.path.join(output_dir, "last.ckpt")
            if os.path.exists(os.path.join(output_dir, "last.ckpt")):
                return last_checkpoint_path

        if checkpoint_path is None:
            checkpoint_path = _determine_checkpoint_to_load(
                os.path.join(self.model_dir, "checkpoints")
            )
        if checkpoint_path is None:
            self.logger.info(
                "No checkpoint found in the model directory. Skipping loading the model."
            )
            return
        self.load_from_checkpoint(checkpoint_path)

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

    def _step_log(self, dict_to_log: Dict[str, float]) -> None:
        """
        Log the value to the logger at the end of the step.
        This writes the values to the progress bar, but not to the log file.

        Args:
            dict_to_log (dict): Dictionary of values to log.
        """
        self.log_dict(
            dict_to_log,
            prog_bar=True,
            on_step=True,
            logger=False,
            on_epoch=False,
            sync_dist=True,
        )

    def _epoch_log(self, dict_to_log: Dict[str, float]) -> None:
        """
        Log the value to the logger at the end of the epoch.
        This writes the values to the progress bar and to the log file.
        """

        self.log_dict(
            dict_to_log,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )
