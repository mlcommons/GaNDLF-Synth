import os

import torch
from torch import nn, optim
from torchvision.utils import save_image

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.stylegan import StyleGan
from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.models.modules.module_abc import SynthesisModule
from torchio.transforms import Resample
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from gandlf_synth.schedulers import get_scheduler

from gandlf_synth.utils.generators import (
    generate_latent_vector,
    get_fixed_latent_vector,
)

from typing import Dict, Union, List, Optional


class UnlabeledStyleGANModule(SynthesisModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: StyleGan
        self.automatic_optimization = False
        self.train_loss_list: List[Dict[float]] = []
        self.alpha = self.model_config.architecture["alpha"]
        self.progressive_epochs = self.model_config.architecture["progressive_epochs"]
        self.current_epoch_in_progressive_epoch = 0
        self.current_step = 0
        self.current_resize_transform = self._determine_current_resize_transform()

    def _determine_current_width_height(self) -> int:
        required_width_height_size = 4 * 2**self.current_step
        return required_width_height_size

    def _determine_resampling_values(self) -> List[int]:
        required_width_height_size = self._determine_current_width_height()
        original_size = self.model_config.tensor_shape[1]
        # for now, the Z-dimension is not resampled
        last_size = (
            self.model_config.tensor_shape[-1]
            if self.model_config.n_dimensions == 3
            else 1
        )
        resampling_value = original_size // required_width_height_size
        return [resampling_value, resampling_value, last_size]

    def _determine_current_resize_transform(self) -> Resample:
        return Resample(self._determine_resampling_values())

    def _resize_to_current_step_demands(self, images: torch.Tensor) -> torch.Tensor:
        images = images.cpu()
        if self.model_config.n_dimensions == 2:
            return torch.stack(
                [
                    self.current_resize_transform(image.unsqueeze(-1)).squeeze(-1)
                    for image in images
                ]
            ).to(self.device)
        elif self.model_config.n_dimensions == 3:
            return torch.stack(
                [self.current_resize_transform(image) for image in images]
            ).to(self.device)

    def _initialize_model(self) -> ModelBase:
        return StyleGan(self.model_config)

    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        disc_loss = get_loss(self.model_config.losses["discriminator"])
        gen_loss = get_loss(self.model_config.losses["generator"])
        return {"disc_loss": disc_loss, "gen_loss": gen_loss}

    @staticmethod
    def _initialize_scheduler(
        disc_optimizer: optim.Optimizer,
        gen_optimizer: optim.Optimizer,
        schedulers_config: dict,
    ) -> Union[Dict[str, Union[optim.lr_scheduler._LRScheduler, None]]]:
        disc_scheduler, gen_scheduler = None, None

        # currently, there is no option of using scheduler for only
        # one of the optimizers. Either need to specify one scheduler for
        # both optimizers or two separate schedulers for each optimizer.

        if "discriminator" in schedulers_config or "generator" in schedulers_config:
            assert (
                "discriminator" in schedulers_config
                and "generator" in schedulers_config
            ), "If you want to use different schedulers for discriminator and generator, you need to specify both."
            disc_scheduler = get_scheduler(
                disc_optimizer, schedulers_config["discriminator"]
            )
            gen_scheduler = get_scheduler(gen_optimizer, schedulers_config["generator"])
        # case when the same scheduler is used for both optimizers
        else:
            disc_scheduler = get_scheduler(disc_optimizer, schedulers_config)
            gen_scheduler = get_scheduler(gen_optimizer, schedulers_config)
        return {"disc_scheduler": disc_scheduler, "gen_scheduler": gen_scheduler}

    def configure_optimizers(self):
        disc_optimizer = get_optimizer(
            model_params=self.model.discriminator.parameters(),
            optimizer_parameters=self.model_config.optimizers["discriminator"],
        )
        gen_optimizer = get_optimizer(
            model_params=self.model.generator.parameters(),
            optimizer_parameters=self.model_config.optimizers["generator"],
        )
        if self.model_config.schedulers is not None:
            schedulers = self._initialize_scheduler(
                disc_optimizer, gen_optimizer, self.model_config.schedulers
            )
            return [disc_optimizer, gen_optimizer], [
                schedulers["disc_scheduler"],
                schedulers["gen_scheduler"],
            ]

        return [disc_optimizer, gen_optimizer]

    def _gradient_penalty(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate gradient penalty for Stylegan that works with both 2D and 3D images.

        Args:
            real: Real images tensor of shape (BATCH_SIZE, C, H, W) or (BATCH_SIZE, C, H, W, D)
            fake: Generated images tensor of same shape as real

        Returns:
            gradient_penalty: Scalar tensor with the gradient penalty
        """
        batch_size = real_images.shape[0]
        channels = real_images.shape[1]
        spatial_dims = real_images.shape[2:]

        beta_shape = (batch_size,) + (1,) * (len(real_images.shape) - 1)
        beta = torch.rand(beta_shape, device=self.device)

        repeat_pattern = [1, channels] + [dim for dim in spatial_dims]
        beta = beta.repeat(*repeat_pattern)

        interpolated_images = real_images * beta + fake_images.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        mixed_scores = self.model.discriminator(
            interpolated_images, self.alpha, self.current_step
        )

        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradient_penalty

    def _generate_latent_vector(self, batch_size: int) -> torch.Tensor:
        latent_vector = torch.randn(
            (batch_size, self.model_config.architecture["latent_vector_size"]),
            device=self.device,
        )
        if self.model_config.n_dimensions == 3:
            latent_vector = latent_vector.unsqueeze(1)
        return latent_vector

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        real_images: torch.Tensor = batch
        real_images = self._resize_to_current_step_demands(real_images)
        gardient_accumulation_steps = self.model_config.accumulate_grad_batches
        gradient_clip_val = self.model_config.gradient_clip_val
        gradient_clip_algorithm = self.model_config.gradient_clip_algorithm

        batch_size = real_images.shape[0]
        latent_vector = self._generate_latent_vector(batch_size)
        loss_disc, loss_gen = self.losses["disc_loss"], self.losses["gen_loss"]
        optimizer_disc, optimizer_gen = self.optimizers()
        fake_images = self.model(latent_vector, self.alpha, self.current_step)
        disc_preds_on_real = self.model.discriminator(
            real_images, self.alpha, self.current_step
        )
        disc_preds_on_fake = self.model.discriminator(
            fake_images.detach(), self.alpha, self.current_step
        )
        self.toggle_optimizer(optimizer_disc)
        gradient_penalty = self._gradient_penalty(real_images, fake_images)
        disc_loss = (
            -(loss_disc(disc_preds_on_real) - loss_disc(disc_preds_on_fake))
            + self.model_config.architecture["gradient_penalty_weight"]
            * gradient_penalty
            + self.model_config.architecture["critic_squared_loss_weight"]
            * torch.mean(disc_preds_on_real**2)
        )
        self.manual_backward(disc_loss)
        self.clip_gradients(optimizer_disc, gradient_clip_val, gradient_clip_algorithm)
        if (batch_idx + 1) % gardient_accumulation_steps == 0:
            optimizer_disc.step()
            optimizer_disc.zero_grad(set_to_none=True)

        self.untoggle_optimizer(optimizer_disc)

        self.toggle_optimizer(optimizer_gen)

        disc_preds_on_fake_gen = self.model.discriminator(
            fake_images, self.alpha, self.current_step
        )
        gen_loss = -loss_gen(disc_preds_on_fake_gen)
        self.manual_backward(gen_loss)
        self.clip_gradients(optimizer_gen, gradient_clip_val, gradient_clip_algorithm)
        if (batch_idx + 1) % gardient_accumulation_steps == 0:
            optimizer_gen.step()
            optimizer_gen.zero_grad(set_to_none=True)

        self.untoggle_optimizer(optimizer_gen)
        self.alpha += batch_size / (
            self.progressive_epochs[self.current_step]
            * len(self.trainer.train_dataloader.dataset)
        )
        self.alpha = min(self.alpha, 1.0)

        loss_dict = {"disc_loss": disc_loss.item(), "gen_loss": gen_loss.item()}
        self._step_log(loss_dict)
        self.train_loss_list.append(loss_dict)

        if self.metric_calculator is not None:
            metric_results = {}
            for metric_name, metric in self.metric_calculator.items():
                metric_results[metric_name] = metric(real_images, fake_images.detach())
            self._step_log(metric_results)

    def validation_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError(
            "Validation step is not implemented for the StyleGAN."
        )

    def test_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("Test step is not implemented for the StyleGAN.")

    def predict_step(self, batch, batch_dx) -> torch.Tensor:
        n_images_to_generate = len(batch)
        latent_vector = generate_latent_vector(
            n_images_to_generate,
            self.model_config.architecture["latent_vector_size"],
            self.model_config.n_dimensions,
            self.device,
        ).squeeze(2, 3)
        fake_images = self.forward(latent_vector)
        if self.postprocessing_transforms is not None:
            for transform in self.postprocessing_transforms:
                fake_images = transform(fake_images)
        fake_images = (fake_images + 1) / 2

        return fake_images

    def forward(
        self,
        latent_vector: torch.Tensor,
        alpha: float = 1.0,
        current_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the unlabeled StyleGAN module. The
        params of generation can be passed as arguments. If absent, they
        will be taken from the module's special attributes.

        Args:
            latent_vector (torch.Tensor): The latent vector to generate images from.
            alpha (float): The alpha value for the progressive growing.
            current_step (int): The current progressive step. If None, the default
                step will be taken from the model config. Useful for inference, where this
                can allow to choose the step (therefore the image size) to generate.
        """
        if current_step is None:
            try:
                current_step = self.model_config.default_forward_step
            except AttributeError:
                raise ValueError(
                    "The current_step argument must be provided if the `default_forward_step` attribute is not set in model config!"
                )
        fake_images = self.model.generator(latent_vector, alpha, current_step)
        return fake_images

    def _update_current_step(self) -> None:
        """
        Check if the given progressive step is finished. If so, update the current step
        and the current resize transform.
        """
        self.current_epoch_in_progressive_epoch += 1
        current_progressive_epochs = self.progressive_epochs[self.current_step]
        if self.current_epoch_in_progressive_epoch >= current_progressive_epochs:
            self.current_epoch_in_progressive_epoch = 0
            self.current_step += 1
            self.current_resize_transform = self._determine_current_resize_transform()

    def _generate_image_set_from_fixed_vector(
        self, n_images_to_generate: int
    ) -> torch.Tensor:
        fixed_latent_vector = get_fixed_latent_vector(
            n_images_to_generate,
            self.model_config.architecture["latent_vector_size"],
            self.model_config.n_dimensions,
            self.device,
            self.model_config.fixed_latent_vector_seed,
        ).squeeze(2, 3)
        temp_alpha = 1.0
        fake_images = self.model.generator(
            fixed_latent_vector, temp_alpha, self.current_step
        )
        return fake_images

    def on_train_epoch_end(self) -> None:
        self._epoch_log(self.train_loss_list)

        process_rank = self.global_rank
        eval_save_interval = self.model_config.save_eval_images_every_n_epochs
        dimension = self.model_config.n_dimensions  # only for 2d
        if (
            dimension == 2
            and eval_save_interval > 0
            and self.current_epoch % eval_save_interval == 0
        ):
            fixed_images_save_path = os.path.join(
                self.model_dir, f"eval_images", f"epoch_{self.current_epoch}"
            )
            if not os.path.exists(fixed_images_save_path):
                os.makedirs(fixed_images_save_path, exist_ok=True)
            last_batch_size = (
                self.model_config.n_fixed_images_to_generate
                % self.model_config.fixed_images_batch_size
            )
            n_batches = (
                self.model_config.n_fixed_images_to_generate
                // self.model_config.fixed_images_batch_size
            )
            if last_batch_size > 0:
                n_batches += 1
            for i in range(n_batches):
                n_images_to_generate = self.model_config.fixed_images_batch_size
                if (i == n_batches - 1) and last_batch_size > 0:
                    n_images_to_generate = last_batch_size
                fake_images = self._generate_image_set_from_fixed_vector(
                    n_images_to_generate
                )
                for n, fake_image in enumerate(fake_images):
                    save_image(
                        fake_image,
                        os.path.join(
                            fixed_images_save_path,
                            f"fake_image_{i*n + n}_{process_rank}.png",
                        ),
                        normalize=True,
                    )
        self._update_current_step()
