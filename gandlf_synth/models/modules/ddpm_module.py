import torch
from torch import nn

from generative.inferers import DiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.ddpm import DDPM
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.utils.compute import backward_pass, perform_parameter_update
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
        noise = torch.randn_like(x, device=x.device)

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

        loss = self.losses(noise_pred.float(), noise.float())

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
        loss_dict = {"loss": loss.detach().item()}
        self.train_loss_list.append(loss_dict)
        self._log_dict(loss_dict)
        # TODO this will not work when the step requires passing loss value
        if self.schedulers is not None:
            self.schedulers.step()
        # for now, we will not calculate metrics for diffusion models in this version as this
        # requires additional generation of samples, slowing down the training

    @torch.no_grad()
    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Validation step is not implemented for the UnlabeledDDPMModule."
        )

    @torch.no_grad()
    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Test step is not implemented for the UnlabeledDDPMModule."
        )

    @torch.no_grad()
    def inference_step(self, **kwargs) -> torch.Tensor:
        n_images_to_generate = kwargs.get("n_images_to_generate", None)
        assert (
            n_images_to_generate is not None
        ), "Number of images to generate is required during the inference pass."

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

    def _on_train_epoch_end(self, epoch: int) -> None:
        avg_loss = sum([loss["loss"] for loss in self.train_loss_list]) / len(
            self.train_loss_list
        )
        self._log(f"Epoch {epoch} loss", avg_loss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _initialize_model(self) -> ModelBase:
        return DDPM(self.model_config)

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
