from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import optim

from logging import Logger
from typing import Dict, Union, Optional, Type

from gandlf_synth.models.configs.config_abc import AbstractModelConfig
from gandlf_synth.models.architectures.base_model import ModelBase


class SynthesisModule(ABC):
    """Abstract class for a synthesis module. It wraps the model architecture
    and logic assocaited with different steps (i.e. forward pass, training_step etc.).
    Greatly inspired by PyTorch Lightning.
    """

    def __init__(
        self,
        model_config: Type[AbstractModelConfig],
        logger: Logger,
        metric_calculator: Optional[object] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the synthesis module.

        Args:
            params (dict): Dictionary of parameters.
            logger (Logger): Logger for logging the values.
            metric_calculator (object,optional): Metric calculator object.
            device (str, optional): Device to perform computations on. Defaults to "cpu".
        """

        super().__init__()
        # This is my idea for now, we can change it later.
        self.model_config = model_config
        self.logger = logger
        self.metric_calculator = metric_calculator
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

    @abstractmethod
    def save_checkpoint(self) -> None:
        """
        Define the logic for saving the model checkpoint.
        It should provide the ability to save the model state, optimizer state and
        other relevant information. Also this will help us to solve the
        multi module problem (i.e. saving generator and discriminator).
        # TODO : Determine the convention for saving the model.
        """
        pass

    @abstractmethod
    def load_checkpoint(self) -> None:
        """
        Define the logic for loading the model checkpoint.
        It should provide the ability to load the model state, optimizer state and
        other relevant information. Also this will help us to solve the
        multi module problem (i.e. loading generator and discriminator).
        """
        pass

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
        self.logger.log(value_name, value_to_log)

    def _log_dict(self, dict_to_log: Dict[str, float]) -> None:
        """
        Log the dictionary of values to the logger.

        Args:
            dict_to_log: Dictionary of values to log.
        """
        for key, value in dict_to_log.items():
            self._log(key, value)

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
