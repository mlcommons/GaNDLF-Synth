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
        self.val_loss_list: List[Dict[float]] = []
        self.test_loss_list: List[Dict[float]] = []

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)
        reconstruction_loss = self.losses(recon_images, x)
        loss = reconstruction_loss + quantization_loss
        backward_pass(
            loss=loss,
            optimizer=self.optimizers,
            model=self.model,
            amp=self.model_config.amp,
            clip_grad=self.model_config.clip_grad,
            clip_mode=self.model_config.clip_mode,
        )

        perform_parameter_update(
            loss=loss, optimizer=self.optimizers, batch_idx=batch_idx
        )
        self.train_loss_list.append(
            {
                "reconstruction_loss": reconstruction_loss.item(),
                "quantization_loss": quantization_loss.item(),
                "total_loss": loss.item(),
            }
        )
        self._log("reconstruction_loss", reconstruction_loss)
        self._log("quantization_loss", quantization_loss)
        # self.
        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                metric_result[metric_name] = metric(recon_images.detach(), x)
            self._log_dict(metric_result)

    @torch.no_grad()
    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)

        reconstruction_loss = self.losses(recon_images, x)

        self._log("val_reconstruction_loss", reconstruction_loss)
        self._log("val_quantization_loss", quantization_loss)
        self.val_loss_list.append(
            {
                "reconstruction_loss": reconstruction_loss.item(),
                "quantization_loss": quantization_loss.item(),
            }
        )

        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                val_metric_name = f"val_{metric_name}"
                metric_result[val_metric_name] = metric(recon_images, x)
            self._log_dict(metric_result)
        # TODO this will not work when the step requires passing loss value
        if self.schedulers is not None:
            self.schedulers.step()

    @torch.no_grad()
    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        recon_images, quantization_loss = self.model(x)
        reconstruction_loss = self.losses(recon_images, x)

        self._log("test_reconstruction_loss", reconstruction_loss)
        self._log("test_quantization_loss", quantization_loss)

        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                inference_metric_name = f"test_{metric_name}"
                metric_result[inference_metric_name] = metric(recon_images, x)
            self._log_dict(metric_result)

    @torch.no_grad()
    def inference_step(self, **kwargs) -> torch.Tensor:
        input_batch = kwargs.get("input_batch")
        assert input_batch is not None, "Input batch is required for inference."
        recon_images, quantization_loss = self.model(input_batch)
        recon_loss = self.losses(recon_images, input_batch)
        self._log("inference_reconstruction_loss", recon_loss)
        self._log("inference_quantization_loss", quantization_loss)
        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                recon_images = transform(recon_images)

        if self.metric_calculator is not None:
            metric_result = {}
            for metric_name, metric in self.metric_calculator.items():
                inference_metric_name = f"inference_{metric_name}"
                metric_result[inference_metric_name] = metric(recon_images, input_batch)
            self._log_dict(metric_result)

        return recon_images

    def _on_train_epoch_end(self, epoch: int) -> None:
        avg_recon_loss = sum(
            [loss["reconstruction_loss"] for loss in self.train_loss_list]
        ) / len(self.train_loss_list)
        avg_quant_loss = sum(
            [loss["quantization_loss"] for loss in self.train_loss_list]
        ) / len(self.train_loss_list)
        avg_total_loss = sum(
            [loss["total_loss"] for loss in self.train_loss_list]
        ) / len(self.train_loss_list)
        self._log(f"Epoch {epoch} reconstruction loss", avg_recon_loss)
        self._log(f"Epoch {epoch} quantization loss", avg_quant_loss)
        self._log(f"Epoch {epoch} total loss", avg_total_loss)

    def _on_validation_epoch_end(self, epoch: int) -> None:
        avg_recon_loss = sum(
            [loss["reconstruction_loss"] for loss in self.val_loss_list]
        ) / len(self.val_loss_list)
        avg_quant_loss = sum(
            [loss["quantization_loss"] for loss in self.val_loss_list]
        ) / len(self.val_loss_list)
        self._log(f"Epoch {epoch} validation reconstruction loss", avg_recon_loss)
        self._log(f"Epoch {epoch} validation quantization loss", avg_quant_loss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _initialize_model(self) -> ModelBase:
        return VQVAE(self.model_config)

    def _initialize_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
        return get_optimizer(
            self.model.parameters(), optimizer_parameters=self.model_config.optimizers
        )

    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        return get_loss(self.model_config.losses)

    def _initialize_schedulers(
        self,
    ) -> Union[
        torch.optim.lr_scheduler._LRScheduler,
        Dict[str, torch.optim.lr_scheduler._LRScheduler],
    ]:
        if self.model_config.schedulers is None:
            return None

        return get_scheduler(scheduler_params=self.model_config.schedulers)
