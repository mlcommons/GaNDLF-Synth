from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import optim

from logging import Logger
from typing import Dict, Union, Optional, Type

from gandlf_synth.models.configs.config_abc import AbstractModelConfig


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
    ) -> None:
        """Initialize the synthesis module.

        Args:
            params (dict): Dictionary of parameters.
            logger (Logger): Logger for logging the values.
            metric_calculator (object,optional): Metric calculator object.
        """

        super().__init__()
        # This is my idea for now, we can change it later.
        self.model_config = model_config
        self.logger = logger
        self.metric_calculator = metric_calculator
        self.optimizers = self._initialize_optimizers()
        self.losses = self._initialize_losses()
        self.schedulers = self._initialize_schedulers()

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
