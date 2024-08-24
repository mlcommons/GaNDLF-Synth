import torch
from torch import nn

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.vqvae import VQVAE
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.utils.compute import backward_pass, perform_parameter_update
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from gandlf_synth.schedulers import get_scheduler

from typing import Dict, Union, List, Tuple


class UnlabeledVQVAEModule(SynthesisModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_loss_list: List[Dict[float]] = []
        self.train_metric_list: List[Dict[float]] = []
        self.val_loss_list: List[Dict[float]] = []
        self.val_metric_list: List[Dict[float]] = []
        self.test_loss_list: List[Dict[float]] = []
        self.test_metric_list: List[Dict[float]] = []

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)
        reconstruction_loss = self.losses(recon_images, x)
        loss = reconstruction_loss + quantization_loss
        loss_dict = {
            "reconstruction_loss": reconstruction_loss.item(),
            "quantization_loss": quantization_loss.item(),
        }
        self.train_loss_list.append(loss_dict)
        self._step_log(loss_dict)
        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                metric_result[metric_name] = metric(recon_images.detach(), x)
            self.train_metric_list.append(metric_result)
            self._step_log(metric_result)
        return loss

    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)

        reconstruction_loss = self.losses(recon_images, x)
        loss = reconstruction_loss + quantization_loss
        loss_dict = {
            "reconstruction_loss": reconstruction_loss.item(),
            "quantization_loss": quantization_loss.item(),
        }

        self.val_loss_list.append(loss_dict)
        self._step_log(loss_dict)

        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                val_metric_name = f"val_{metric_name}"
                metric_result[val_metric_name] = metric(recon_images, x)
            self._step_log(metric_result)
            self.val_metric_list.append(metric_result)
        return loss

    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)
        reconstruction_loss = self.losses(recon_images, x)
        loss_dict = {
            "reconstruction_loss": reconstruction_loss.item(),
            "quantization_loss": quantization_loss.item(),
        }
        self.test_loss_list.append(loss_dict)
        self._step_log(loss_dict)
        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                inference_metric_name = f"test_{metric_name}"
                metric_result[inference_metric_name] = metric(recon_images, x)
            self._step_log(metric_result)
            self.test_metric_list.append(metric_result)

    @torch.no_grad()
    def inference_step(self, **kwargs) -> torch.Tensor:
        input_batch = kwargs.get("input_batch")
        assert input_batch is not None, "Input batch is required for inference."
        recon_images, quantization_loss = self.model(input_batch)
        recon_loss = self.losses(recon_images, input_batch)
        # self._step_log("inference_reconstruction_loss", recon_loss)
        # self._step_log("inference_quantization_loss", quantization_loss)
        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                recon_images = transform(recon_images)

        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                inference_metric_name = f"inference_{metric_name}"
                metric_result[inference_metric_name] = metric(recon_images, input_batch)
            # self._step_log(metric_result)

        return recon_images

    def on_train_epoch_end(self) -> None:
        self._epoch_log(self.train_loss_list)
        self._epoch_log(self.train_metric_list)

    def on_validation_epoch_end(self) -> None:
        self._epoch_log(self.val_loss_list)
        self._epoch_log(self.val_metric_list)

    def on_test_epoch_end(self) -> None:
        self._epoch_log(self.test_loss_list)
        self._epoch_log(self.test_metric_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _initialize_model(self) -> ModelBase:
        return VQVAE(self.model_config)

    def configure_optimizers(self):
        return get_optimizer(
            self.model.parameters(), optimizer_parameters=self.model_config.optimizers
        )

    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        return get_loss(self.model_config.losses)
