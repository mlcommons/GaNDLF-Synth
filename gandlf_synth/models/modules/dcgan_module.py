import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.architectures.dcgan import DCGAN
from module_abc import SynthesisModule
from utils.compute import backward_pass
from utils.generators import generate_latent_vector, get_fixed_latent_vector
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import get_loss
from typing import Dict, Union


# TODO
class UnlabeledDCGANModule(SynthesisModule):
    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        real_images = self._ensure_device_placement(batch)
        batch_size = real_images.size(0)
        label_real = torch.full(
            (batch_size,), fill_value=1.0, dtype=torch.float, device=self.device
        )
        preds_real = self.model.discriminator(real_images)
        disc_loss_real = self.losses["disc_loss"](preds_real, label_real)

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
            model_params=self.model.discriminator.parameters,
            optimizer_parameters=self.model_config.optimizers["discriminator"],
        )
        gen_optimizer = get_optimizer(
            model_params=self.model.generator.parameters,
            optimizer_parameters=self.model_config.optimizers["generator"],
        )
        return {"disc_optimizer": disc_optimizer, "gen_optimizer": gen_optimizer}

    # TODO Not really that important now
    def _initialize_schedulers(
        self,
    ) -> Union[
        torch.optim.lr_scheduler._LRScheduler,
        Dict[str, torch.optim.lr_scheduler._LRScheduler],
        None,
    ]:
        print("Initializing schedulers")
        return None

    # TODO
    def save_checkpoint(self) -> None:
        print("Saving checkpoint!")

    # TODO
    def load_checkpoint(self) -> None:
        print("Loading checkpoint!")
