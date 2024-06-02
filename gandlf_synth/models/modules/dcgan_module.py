import torch
from torch import nn

from module_abc import SynthesisModule
from utils.compute import backward_pass
from gandlf_synth.optimizers import global_optimizer_dict
from gandlf_synth.losses import global_losses_dict
from typing import Dict, Union


class UnlabeledDCGANModule(SynthesisModule):
    def _initialize_losses(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        self.disc_loss = global_losses_dict[self.model_config.disc_loss]()
        self.gen_loss = global_losses_dict[self.model_config.gen_loss]()
        return {"disc_loss": self.disc_loss, "gen_loss": self.gen_loss}
    def 