from gandlf_synth.models.modules.module_abc import SynthesisModule
from gandlf_synth.models.modules.dcgan_module import UnlabeledDCGANModule
from gandlf_synth.models.modules.vqvae_module import UnlabeledVQVAEModule
from gandlf_synth.models.modules.ddpm_module import UnlabeledDDPMModule
from gandlf_synth.models.modules.stylegan_module import UnlabeledStyleGANModule
from gandlf_synth.models.configs.config_abc import AbstractModelConfig

from typing import Type, Optional, Dict, List, Callable


class ModuleFactory:
    AVAILABE_MODULES = {
        "unlabeled_dcgan": UnlabeledDCGANModule,
        "unlabeled_vqvae": UnlabeledVQVAEModule,
        "unlabeled_ddpm": UnlabeledDDPMModule,
        "unlabeled_stylegan": UnlabeledStyleGANModule,
    }
    """
    Class responsible for creating modules.
    """

    def __init__(
        self,
        model_config: Type[AbstractModelConfig],
        model_dir: str,
        metric_calculator: Optional[Dict[str, object]] = None,
        postprocessing_transforms: Optional[List[Callable]] = None,
    ):
        """
        Initialize the ModuleFactory.

        Args:
            model_config (Type[AbstractModelConfig]): The model configuration object.
            logger (Logger): The logger object.
            model_dir (str): Model and results output directory.
            metric_calculator (dict, optional): The metric calculator dictionary. Defaults to None.
            postprocessing_transforms (List[Callable], optional): The postprocessing transformations to apply. Defaults to None.
            device (str, optional): The device to perform computations on. Defaults to "cpu".
        """

        self.model_config = model_config
        self.model_dir = model_dir
        self.metric_calculator = metric_calculator
        self.postprocessing_transforms = postprocessing_transforms

    def _parse_module_name(self) -> str:
        """
        Method to parse the module name from the model config.

        Returns:
            str: The module name.
        """
        model_name = self.model_config.model_name
        labeling_paradigm = self.model_config.labeling_paradigm
        return f"{labeling_paradigm}_{model_name}"

    def get_module(self) -> Type[SynthesisModule]:
        """
        Method to get the module based on the model config.

        Returns:
            Module (SynthesisModule): The synthesis module object.
        """

        module_name = self._parse_module_name()
        assert module_name in self.AVAILABE_MODULES.keys(), (
            f"Module {module_name} not found. "
            f"Available modules: {self.AVAILABE_MODULES.keys()}"
        )
        return self.AVAILABE_MODULES[module_name](
            model_config=self.model_config,
            model_dir=self.model_dir,
            metric_calculator=self.metric_calculator,
            postprocessing_transforms=self.postprocessing_transforms,
        )
