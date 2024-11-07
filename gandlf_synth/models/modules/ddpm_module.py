import torch
from torch import nn

from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.ddpm import DDPM
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from gandlf_synth.schedulers import get_scheduler

from typing import Dict, Union, List


class UnlabeledDDPMModule(SynthesisModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_loss_list: List[Dict[float]] = []
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.model_config.architecture["num_train_timesteps"]
        )
        self.inferer = DiffusionInferer(self.scheduler)

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        x = batch
        noise = torch.randn_like(x).type_as(x)
        # Create timesteps
        timesteps = torch.randint(
            0,
            self.inferer.scheduler.num_train_timesteps,
            (x.shape[0],),
            device=x.device,
        ).long()
        noise_pred = self.inferer(
            inputs=x, diffusion_model=self.model, noise=noise, timesteps=timesteps
        )
        loss = self.losses(noise_pred, noise)

        loss_dict = {"loss": loss.detach().item()}
        self.train_loss_list.append(loss_dict)
        self._step_log(loss_dict)
        return loss
        # for now, we will not calculate metrics for diffusion models in this version as this
        # requires additional generation of samples, slowing down the training

    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Validation step is not implemented for the UnlabeledDDPMModule."
        )

    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Test step is not implemented for the UnlabeledDDPMModule."
        )

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        # Batch is a set of batch_idxes, representing separate samples
        # to generate, for example batch=torch.Tensor([0, 1, 2, 3, 4])
        n_images_to_generate = len(batch)

        noise = torch.randn(
            n_images_to_generate,
            self.model_config.n_channels,
            *self.model_config.tensor_shape,
            device=self.device,
        )
        self.scheduler.set_timesteps(
            num_inference_steps=self.model_config.architecture["num_eval_timesteps"]
        )
        generated_images = self.inferer.sample(
            input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler
        )
        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                generated_images = transform(generated_images)

        return generated_images

    def on_train_epoch_end(self) -> None:
        self._epoch_log(self.train_loss_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _initialize_model(self) -> ModelBase:
        return DDPM(self.model_config)

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
