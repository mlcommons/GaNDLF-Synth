import torch
from torch import nn

from module_abc import SynthesisModule
from utils.compute import backward_pass
from gandlf_synth.optimizers import get_optimizer
from gandlf_synth.losses import global_losses_dict
from typing import Dict, Union


class UnlabeledDCGANModule(SynthesisModule):
    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        self.disc_loss = global_losses_dict[self.model_config.disc_loss]()
        self.gen_loss = global_losses_dict[self.model_config.gen_loss]()
        return {"disc_loss": self.disc_loss, "gen_loss": self.gen_loss}

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
