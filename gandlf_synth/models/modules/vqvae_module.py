import torch
from torch import nn

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.vqvae import VQVAE
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from gandlf_synth.schedulers import get_scheduler


from typing import Dict, Union


class UnlabeledVQVAEModule(SynthesisModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # TODO how to display the predict metrics/results? It cannot be logged
        self.phase_loss_lists = {"train": [], "val": [], "test": [], "predict": []}
        self.phase_metric_lists = {"train": [], "val": [], "test": [], "predict": []}

    def _calculate_and_log_metrics(
        self, recon_images: torch.Tensor, x: torch.Tensor, phase: str
    ):
        metric_result = {}
        for metric_name, metric in self.metric_calculator.items():
            if phase != "train":
                metric_name = f"{phase}_{metric_name}"
            metric_result[metric_name] = metric(recon_images, x)
        self.phase_metric_lists[phase].append(metric_result)
        if phase != "predict":
            self._step_log(metric_result)

    def _common_step(self, batch: object, phase: str) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)
        reconstruction_loss = self.losses(recon_images, x)
        loss = reconstruction_loss + quantization_loss

        loss_dict = {
            "reconstruction_loss": reconstruction_loss.item(),
            "quantization_loss": quantization_loss.item(),
        }
        self.phase_loss_lists[phase].append(loss_dict)
        if phase != "predict":
            self._step_log(loss_dict)
        if self.metric_calculator is not None:
            self._calculate_and_log_metrics(recon_images, x, phase)

        return loss, recon_images

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        loss, _ = self._common_step(batch, "train")
        return loss

    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        loss, _ = self._common_step(batch, "val")
        return loss

    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        self._common_step(batch, "test")

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        _, recon_images = self._common_step(batch, "predict")

        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                recon_images = transform(recon_images)

        return recon_images

    def on_train_epoch_end(self) -> None:
        self._epoch_log(self.phase_loss_lists["train"])
        self._epoch_log(self.phase_metric_lists["train"])

    def on_validation_epoch_end(self) -> None:
        self._epoch_log(self.phase_loss_lists["val"])
        self._epoch_log(self.phase_metric_lists["val"])

    def on_test_epoch_end(self) -> None:
        self._epoch_log(self.phase_loss_lists["test"])
        self._epoch_log(self.phase_metric_lists["test"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _initialize_model(self) -> ModelBase:
        return VQVAE(self.model_config)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.model.parameters(), optimizer_parameters=self.model_config.optimizers
        )
        if self.model_config.schedulers is not None:
            scheduler = get_scheduler(
                optimizer, scheduler_params=self.model_config.schedulers
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        return get_loss(self.model_config.losses)
