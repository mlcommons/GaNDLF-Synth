import os
from pathlib import Path
from argparse import ArgumentParser

import gandlf_synth

from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.models.configs.model_config_factory import ModelConfigFactory
from gandlf_synth.data.datasets import DatasetFactory
from gandlf_synth.data.dataloaders import DataloaderFactory
from gandlf_synth.models.modules.dcgan_module import UnlabeledDCGANModule


TEST_DIR = Path(__file__).parent.absolute().__str__()

TEST_CONFIG_PATH = os.path.join(TEST_DIR, "syntheis_module_config.yaml")
CSV_PATH = os.path.join(os.path.dirname(TEST_DIR), "unlabeled_data.csv")


def main():
    # load the configuration
    config_manager = ConfigManager(TEST_CONFIG_PATH)

    global_config, model_config = config_manager.prepare_configs()

    dataset_factory = DatasetFactory()
    dataloader_factory = DataloaderFactory(global_config)

    dataset = dataset_factory.get_dataset(
        CSV_PATH, None, model_config.labeling_paradigm
    )
    dataloader = dataloader_factory.get_training_dataloader(dataset)
    print(dataset[0])


if __name__ == "__main__":
    main()
