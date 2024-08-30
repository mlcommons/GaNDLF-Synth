import os
import inspect
import logging
from pathlib import Path

import pandas as pd
from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.training_manager import TrainingManager
from testing.testing_utils import ContextManagerTests

TEST_DIR = Path(__file__).parent.absolute().__str__()
OUTPUT_DIR = os.path.join(TEST_DIR, "output")
INFERENCE_OUTPUT_DIR = os.path.join(TEST_DIR, "inference_output")
# we test on vqvae model as it allows to use all functionalities for both train/val/test
CONFIG_PATH = os.path.join(
    os.path.dirname(TEST_DIR), "configs", "module_config_vqvae.yaml"
)
LOG_DIR = os.path.join(TEST_DIR, "logs")

# we test on unlabeled 2d_rad, our main goal is to check the training manager functionality
CSV_PATH = os.path.join(
    os.path.dirname(TEST_DIR), "data", "2d_rad", "2d_rad_unlabeled_data.csv"
)
DEVICE = "cpu"
BASIC_LOGGER_CONFIG = logging.basicConfig(
    filename=f"{LOG_DIR}/synthesis_module_tests.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level="INFO",
)
LOGGER_OBJECT = logging.getLogger("synthesis_module_logger")


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

GLOBAL_CONFIG, MODEL_CONFIG = ConfigManager(CONFIG_PATH).prepare_configs()
EXAMPLE_DATAFRAME = pd.read_csv(CSV_PATH)

# TODO: This test is checking the pipeline created manually, wtihout encampsulating it in
# a training manager. For now it is commented out, as the same logic happens in training manager
# in the future we may remove it or replace it with some modification.


# def test_initial_pipeline_module():
#     test_name = inspect.currentframe().f_code.co_name
#     with ContextManagerTests(
#         test_dir=TEST_DIR, test_name=test_name, output_dir=OUTPUT_DIR
#     ):
#         for module in AVAILABLE_MODULES:
#             # labeling_paradigm, model_name = parse_available_module(module)
#             with open(TEST_CONFIG_PATH, "r") as config_file:
#                 config = yaml.safe_load(config_file)
#                 # config["model_config"]["model_name"] = model_name
#                 # config["model_config"]["labeling_paradigm"] = labeling_paradigm
#             with open(TEST_CONFIG_PATH, "w") as config_file:
#                 yaml.dump(config, config_file)
#             config_manager = ConfigManager(TEST_CONFIG_PATH)

#             GLOBAL_CONFIG, model_config = config_manager.prepare_configs()
#             # TODO this needs to be replaced with proper transforms
#             RESIZE_TRANSFORM = Compose([Resize((128, 128, 1))])
#             dataset_factory = DatasetFactory()
#             dataloader_factory = DataloaderFactory(global_config)
#             EXAMPLE_DATAFRAME = pd.read_csv(CSV_PATH)
#             dataset = dataset_factory.get_dataset(
#                 EXAMPLE_DATAFRAME, RESIZE_TRANSFORM, model_config.labeling_paradigm
#             )

#             dataloader = dataloader_factory.get_training_dataloader(dataset)

#             module_factory = ModuleFactory(
#                 model_config=MODEL_CONFIG,
#                 logger=LOGGER_OBJECT,
#                 metric_calculator=None,
#                 model_dir=OUTPUT_DIR,
#             )
#             module = module_factory.get_module()

#             trainer = pl.Trainer(max_epochs=1)
#             trainer.fit(module, dataloader)


#             for batch_idx, batch in enumerate(dataloader):
#                 module.training_step(batch, batch_idx)
#                 print("Training step completed!")
#                 break


def test_training_manager_val_test_df():
    """
    Test with val and test dataframes provided for splitting the data.
    """
    test_name = inspect.currentframe().f_code.co_name
    with ContextManagerTests(
        test_dir=TEST_DIR, test_name=test_name, output_dir=OUTPUT_DIR
    ):
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=False,
            reset=False,
            val_dataframe=EXAMPLE_DATAFRAME,
            test_dataframe=EXAMPLE_DATAFRAME,
        )
        training_manager.run_training()


def test_training_manager_val_test_ratio():
    """
    Test with val and test ratio provided for splitting the data.
    """
    test_name = inspect.currentframe().f_code.co_name
    with ContextManagerTests(
        test_dir=TEST_DIR, test_name=test_name, output_dir=OUTPUT_DIR
    ):
        config_manager = ConfigManager(CONFIG_PATH)
        EXAMPLE_DATAFRAME = pd.read_csv(CSV_PATH)
        GLOBAL_CONFIG, model_config = config_manager.prepare_configs()
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=False,
            reset=False,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        training_manager.run_training()


def test_training_manager_val_test_fallback():
    """
    Test fallback to dataframes when both ratios and dataframes are provided.
    Should fallback to dataframes.
    """
    test_name = inspect.currentframe().f_code.co_name
    with ContextManagerTests(
        test_dir=TEST_DIR, test_name=test_name, output_dir=OUTPUT_DIR
    ):
        config_manager = ConfigManager(CONFIG_PATH)
        GLOBAL_CONFIG, model_config = config_manager.prepare_configs()
        EXAMPLE_DATAFRAME = pd.read_csv(CSV_PATH)
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=False,
            reset=False,
            val_ratio=0.1,
            test_ratio=0.1,
            val_dataframe=EXAMPLE_DATAFRAME,
            test_dataframe=EXAMPLE_DATAFRAME,
        )
        training_manager.run_training()


def test_training_manager_reset_resume():
    """
    Test resetting and resuming training.
    """
    test_name = inspect.currentframe().f_code.co_name
    with ContextManagerTests(
        test_dir=TEST_DIR, test_name=test_name, output_dir=OUTPUT_DIR
    ):
        # Test resetting and resuming
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=False,
            reset=False,
        )
        training_manager.run_training()
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=False,
            reset=True,
        )
        training_manager.run_training()
        training_manager = TrainingManager(
            train_dataframe=EXAMPLE_DATAFRAME,
            output_dir=OUTPUT_DIR,
            global_config=GLOBAL_CONFIG,
            model_config=MODEL_CONFIG,
            resume=True,
            reset=False,
        )
        training_manager.run_training()
