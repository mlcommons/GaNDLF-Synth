import os
import logging
import yaml
from pathlib import Path
from copy import deepcopy

import pandas as pd
from torchio.transforms import Compose, Resize

from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.data.datasets_factory import DatasetFactory
from gandlf_synth.data.dataloaders_factory import DataloaderFactory
from gandlf_synth.models.modules.module_factory import ModuleFactory
from gandlf_synth.training_manager import TrainingManager
from typing import List

TEST_DIR = Path(__file__).parent.absolute().__str__()

TEST_CONFIG_PATH = os.path.join(TEST_DIR, "syntheis_module_config.yaml")
with open(TEST_CONFIG_PATH, "r") as config_file:
    ORIGINAL_CONFIG = yaml.safe_load(config_file)
CSV_PATH = os.path.join(os.path.dirname(TEST_DIR), "unlabeled_data.csv")
DEVICE = "cpu"
LOGGER_OBJECT = logging.Logger("synthesis_module_logger", level=logging.DEBUG)

# Take all available modules registered
AVAILABLE_MODULES = list(ModuleFactory.AVAILABE_MODULES.keys())
# Take all available model configs registered
AVAILABLE_CONFIGS = list(ModuleFactory.AVAILABE_MODULES.keys())


def restore_config():
    """
    Sanitizing function to restore the original config file after the tests are done, in
    case it is overwritten.
    """

    with open(TEST_CONFIG_PATH, "w") as config_file:
        yaml.dump(ORIGINAL_CONFIG, config_file)


class ContextManagerTests:
    """
    Context manager ensuring that certain operations are performed before and after the tests.
    """

    def __enter__(self):
        """
        Method to be executed before the tests.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method to be executed after the tests.
        """
        # Later we may move output dir sanitization here too, and other stuff
        restore_config()


def parse_available_module(module_name: str) -> List[str]:
    """
    Helper method to parse the module name into its components (labeling paradigm and model name).
    Used to replace the model name and labeling paradigm in the config file to check all
    available modules and configs in one go.

    Args:
        module_name (str): The module name from available modules.

    Returns:
        labeling_paradigm (str): The labeling paradigm.
        model_name (str): The model name.

    """
    return module_name.split("_")


def test_module_config_pairs():
    # Check if all available modules have a corresponding config and vice versa
    for module in AVAILABLE_MODULES:
        assert (
            module in AVAILABLE_CONFIGS
        ), f"Module {module} does not have a corresponding config"
    for config in AVAILABLE_CONFIGS:
        assert (
            config in AVAILABLE_MODULES
        ), f"Config {config} does not have a corresponding module"


# def test_initial_pipeline_module():
#     with ContextManagerTests():
#         for module in AVAILABLE_MODULES:
#             labeling_paradigm, model_name = parse_available_module(module)
#             with open(TEST_CONFIG_PATH, "r") as config_file:
#                 config = yaml.safe_load(config_file)
#                 config["model_config"]["model_name"] = model_name
#                 config["model_config"]["labeling_paradigm"] = labeling_paradigm
#             with open(TEST_CONFIG_PATH, "w") as config_file:
#                 yaml.dump(config, config_file)
#             config_manager = ConfigManager(TEST_CONFIG_PATH)

#             global_config, model_config = config_manager.prepare_configs()
#             # TODO this needs to be replaced with proper transforms
#             RESIZE_TRANSFORM = Compose([Resize((128, 128, 1))])
#             dataset_factory = DatasetFactory()
#             dataloader_factory = DataloaderFactory(global_config)
#             example_dataframe = pd.read_csv(CSV_PATH)
#             dataset = dataset_factory.get_dataset(
#                 example_dataframe, RESIZE_TRANSFORM, model_config.labeling_paradigm
#             )

#             dataloader = dataloader_factory.get_training_dataloader(dataset)

#             module_factory = ModuleFactory(
#                 model_config=model_config,
#                 logger=LOGGER_OBJECT,
#                 metric_calculator=None,
#                 device=DEVICE,
#             )
#             module = module_factory.get_module()

#             for batch_idx, batch in enumerate(dataloader):
#                 module.training_step(batch, batch_idx)
#                 print("Training step completed!")
#                 break


def test_training_manager():
    config_manager = ConfigManager(TEST_CONFIG_PATH)
    global_config, model_config = config_manager.prepare_configs()
    output_dir = os.path.join(TEST_DIR, "output")
    example_dataframe = pd.read_csv(CSV_PATH)
    # test the run when there is no validation and testing data
    training_manager = TrainingManager(
        train_dataframe=example_dataframe,
        output_dir=output_dir,
        global_config=global_config,
        model_config=model_config,
        resume=False,
        reset=False,
        device=DEVICE,
    )
    training_manager.run_training()
