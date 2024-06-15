import warnings

import torch
from torch import nn


from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.dcgan import DCGAN
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.utils.compute import backward_pass
from gandlf_synth.utils.generators import (
    generate_latent_vector,
    get_fixed_latent_vector,
)
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from gandlf_synth.schedulers import get_scheduler

from typing import Dict, Union


# TODO
class UnlabeledDCGANModule(SynthesisModule):
    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        real_images = self._ensure_device_placement(batch)
        batch_size = real_images.shape[0]
        latent_vector = generate_latent_vector(
            batch_size,
            self.model_config.architecture["latent_vector_size"],
            self.model_config.n_dimensions,
            self.device,
        )
        # DISCRIMINATOR PASS WITH REAL IMAGES
        self.optimizers["disc_optimizer"].zero_grad(set_to_none=True)
        label_real = torch.full(
            (batch_size, 1), fill_value=1.0, dtype=torch.float, device=self.device
        )
        preds_real = self.model.discriminator(real_images)
        disc_loss_real = self.losses["disc_loss"](preds_real, label_real)
        backward_pass(
            loss=disc_loss_real,
            optimizer=self.optimizers["disc_optimizer"],
            model=self.model.discriminator,
            amp=self.model_config.amp,
            clip_grad=self.model_config.clip_grad,
            clip_mode=self.model_config.clip_mode,
        )

        # DISCRIMINATOR PASS WITH FAKE IMAGES
        label_fake = label_real.fill_(0.0)  # swap the labels for fake images
        fake_images = self.model.generator(latent_vector)

        preds_fake = self.model.discriminator(fake_images.detach())
        disc_loss_fake = self.losses["disc_loss"](preds_fake, label_fake)
        backward_pass(
            loss=disc_loss_fake,
            optimizer=self.optimizers["disc_optimizer"],
            model=self.model.discriminator,
            amp=self.model_config.amp,
            clip_grad=self.model_config.clip_grad,
            clip_mode=self.model_config.clip_mode,
        )
        is_any_nan = torch.isnan(disc_loss_real) or torch.isnan(disc_loss_fake)
        if not is_any_nan:
            # is this correct? is it propagating correctly?
            loss_disc = disc_loss_real + disc_loss_fake
            self.optimizers["disc_optimizer"].step()
            self.optimizers["disc_optimizer"].zero_grad(set_to_none=True)
        else:
            warnings.warn(
                f"NaN loss detected in discriminator step for batch {batch_idx}, the step will be skipped",
                RuntimeWarning,
            )
        # Scheduler step
        if self.schedulers["disc_scheduler"] is not None:
            self.schedulers["disc_scheduler"].step()

        # GENERATOR PASS
        self.optimizers["gen_optimizer"].zero_grad(set_to_none=True)
        label_real.fill_(1.0)  # swap the labels for fake images
        preds_fake = self.model.discriminator(fake_images)
        gen_loss = self.losses["gen_loss"](preds_fake, label_real)
        backward_pass(
            loss=gen_loss,
            optimizer=self.optimizers["gen_optimizer"],
            model=self.model.generator,
            amp=self.model_config.amp,
            clip_grad=self.model_config.clip_grad,
            clip_mode=self.model_config.clip_mode,
        )
        is_any_nan = torch.isnan(gen_loss)
        if not is_any_nan:
            self.optimizers["gen_optimizer"].step()
            self.optimizers["gen_optimizer"].zero_grad(set_to_none=True)
        else:
            warnings.warn(
                f"NaN loss detected in generator step for batch {batch_idx}, the step will be skipped",
                RuntimeWarning,
            )

        # Scheduler step
        if self.schedulers["gen_scheduler"] is not None:
            self.schedulers["gen_scheduler"].step()

        self._log("disc_loss", loss_disc)
        self._log("gen_loss", gen_loss)
        if self.metric_calculator is not None:
            metric_results = {}
            for metric_name, metric in self.metric_calculator.items():
                metric_results[metric_name] = metric(real_images, fake_images.detach())
            self._log_dict(metric_results)

    @torch.no_grad
    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        pass

    # TODO
    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        print("Test step")

    # TODO
    def inference_step(self, **kwargs) -> torch.Tensor:
        print("Inference step")
        n_images_to_generate = kwargs.get("n_images_to_generate", None)
        assert (
            n_images_to_generate is not None
        ), "Number of images to generate is required during the inference pass."

        fake_images = self.forward(n_images_to_generate=n_images_to_generate)
        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                fake_images = transform(fake_images)
        return fake_images

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass of the unlabeled DCGAN module. This method is considered
        a call to a generator to produce given number of images.

        Args:
            n_images_to_generate (int): Number of images to generate.
        """
        n_images_to_generate = kwargs.get("n_images_to_generate", None)
        assert (
            n_images_to_generate is not None
        ), "Number of images to generate is required during the forward pass."

        latent_vector = generate_latent_vector(
            n_images_to_generate,
            self.model_config.architecture["latent_vector_size"],
            self.model_config.n_dimensions,
            self.device,
        )
        fake_images = self.model.generator(latent_vector)
        return fake_images

    def _initialize_model(self) -> ModelBase:
        return DCGAN(self.model_config)

    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        disc_loss = get_loss(self.model_config.losses["discriminator"])
        gen_loss = get_loss(self.model_config.losses["generator"])
        return {"disc_loss": disc_loss, "gen_loss": gen_loss}

    def _initialize_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
        disc_optimizer = get_optimizer(
            model_params=self.model.discriminator.parameters(),
            optimizer_parameters=self.model_config.optimizers["discriminator"],
        )
        gen_optimizer = get_optimizer(
            model_params=self.model.generator.parameters(),
            optimizer_parameters=self.model_config.optimizers["generator"],
        )
        return {"disc_optimizer": disc_optimizer, "gen_optimizer": gen_optimizer}

    def _initialize_schedulers(
        self,
    ) -> Union[
        torch.optim.lr_scheduler._LRScheduler,
        Dict[str, torch.optim.lr_scheduler._LRScheduler],
        None,
    ]:
        # For more info on scheduler parsing check the get_scheduler function in the schedulers/__init__.py
        UNSUPPORTED_SCHEDULERS = [
            "reduceonplateau",
            "plateau",
            "reduce-on-plateau",
            "reduce_on_plateau",
        ]
        disc_scheduler = None
        gen_scheduler = None
        if self.model_config.schedulers is not None:
            if "discriminator" in self.model_config.schedulers:
                disc_scheduler_config = self.model_config.schedulers["discriminator"]
                assert (
                    disc_scheduler_config not in UNSUPPORTED_SCHEDULERS
                ), f"Scheduler {disc_scheduler_config.keys()[0]} is not supported for the DCGAN model."
                disc_scheduler = get_scheduler(disc_scheduler_config)
            if "generator" in self.model_config.schedulers:
                gen_scheduler_config = self.model_config.schedulers["generator"]
                assert (
                    gen_scheduler_config not in UNSUPPORTED_SCHEDULERS
                ), f"Scheduler {gen_scheduler_config.keys()[0]} is not supported for the DCGAN model."
                gen_scheduler = get_scheduler(gen_scheduler_config)
        return {"disc_scheduler": disc_scheduler, "gen_scheduler": gen_scheduler}
