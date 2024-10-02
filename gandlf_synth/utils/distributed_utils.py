from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.strategy import Strategy as LightningStrategy
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from torch.cuda import device_count

from typing import Union


class DistributedStrategyFactory:
    """
    Class responsible for creating the distributed strategy based on the input parameters.
    """

    SUPPORTED_STRATEGIES = {
        "auto": "auto",
        "ddp": DDPStrategy,
        "deepspeed": DeepSpeedStrategy,
    }

    def __init__(self, global_config: dict):
        """
        Initalize the factory with the global config parameters.

        Args:
            global_config (dict): The global configuration parameters.
        """
        self.global_config = global_config
        self.gpu_devices = device_count()
        self._parse_compute_parameters()

    def _parse_compute_parameters(self):
        """
        Sets the attributes based on the global config parameters.
        """
        compute_config: dict = self.global_config.get("compute", {})
        self.strategy_name = str(compute_config.get("strategy", "auto")).lower()

        assert (
            self.strategy_name in self.SUPPORTED_STRATEGIES.keys()
        ), f"Strategy {self.strategy_name} is not supported. Allowed strategies: {self.SUPPORTED_STRATEGIES}. You can also set 'auto' to let the trainer choose the strategy."

        self.strategy_config: dict = compute_config.get("strategy_config", {})
        if self.strategy_name == "deepseed":
            assert self.strategy_config.get(
                "config"
            ), "DeepSpeed requires a 'config' to be set in the 'strategy_config' to properly use the strategy. This should point to json with the DeepSpeed configuration."

    def get_strategy(self) -> Union[LightningStrategy, str]:
        """
        Returns the distributed strategy based on the global config parameters.
        In case no compute specification is available, it will return an
        "auto" string to let the trainer define the strategy itself.
        """

        if self.strategy_name == "auto":
            return "auto"

        return self.SUPPORTED_STRATEGIES[self.strategy_name](**self.strategy_config)
