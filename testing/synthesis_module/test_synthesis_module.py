import os
import logging

from pathlib import Path
from argparse import ArgumentParser

from torchio.transforms import Compose, Resize

from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.models.configs.model_config_factory import ModelConfigFactory
from gandlf_synth.data.datasets import DatasetFactory
from gandlf_synth.data.dataloaders import DataloaderFactory
from gandlf_synth.models.modules.dcgan_module import UnlabeledDCGANModule


TEST_DIR = Path(__file__).parent.absolute().__str__()

TEST_CONFIG_PATH = os.path.join(TEST_DIR, "syntheis_module_config.yaml")
CSV_PATH = os.path.join(os.path.dirname(TEST_DIR), "unlabeled_data.csv")
DEVICE = "cpu"
LOGGER_OBJECT = logging.Logger("synthesis_module_logger", level=logging.DEBUG)


def main():
    # load the configuration

    config_manager = ConfigManager(TEST_CONFIG_PATH)

    global_config, model_config = config_manager.prepare_configs()
    RESIZE_TRANSFORM = Compose([Resize((128, 128, 1))])
    dataset_factory = DatasetFactory()
    dataloader_factory = DataloaderFactory(global_config)

    dataset = dataset_factory.get_dataset(
        CSV_PATH, RESIZE_TRANSFORM, model_config.labeling_paradigm
    )
    dataloader = dataloader_factory.get_training_dataloader(dataset)
    module = UnlabeledDCGANModule(
        model_config=model_config,
        logger=LOGGER_OBJECT,
        metric_calculator=None,
        device=DEVICE,
    )
    for batch_idx, batch in enumerate(dataloader):
        module.training_step(batch, batch_idx)
        print("Training step completed!")
        break


if __name__ == "__main__":
    main()
