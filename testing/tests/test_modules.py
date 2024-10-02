import os
import inspect
import logging
from pathlib import Path
import pytest
import pandas as pd
from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.training_manager import TrainingManager
from gandlf_synth.inference_manager import InferenceManager
from testing.testing_utils import (
    ContextManagerTests,
    set_3d_dataloader_resize,
    set_input_tensor_shapes_to_3d,
    create_csv_modality_labeling_type_path,
)

TEST_DIR = Path(__file__).parent.absolute().__str__()
OUTPUT_DIR = os.path.join(TEST_DIR, "output")
INFERENCE_OUTPUT_DIR = os.path.join(TEST_DIR, "inference_output")
LOG_DIR = os.path.join(TEST_DIR, "logs")
GENERAL_DATA_DIR = os.path.join(os.path.dirname(TEST_DIR), "data")
LABELING_TYPES = ["unlabeled"]


def setup_logging():
    logging.basicConfig(
        filename=f"{LOG_DIR}/synthesis_module_tests.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
    )
    return logging.getLogger("synthesis_module_logger")


LOGGER_OBJECT = setup_logging()


def run_test(config_path, modality, n_dimensions, labeling_type, is_histo=False):
    test_name = inspect.currentframe().f_code.co_name
    config_manager = ConfigManager(config_path)
    global_config, model_config = config_manager.prepare_configs()

    global_config["modality"] = "histo" if is_histo else "rad"
    model_config.n_dimensions = n_dimensions
    model_config.labeling_paradigm = labeling_type

    if n_dimensions == 3:
        set_3d_dataloader_resize(global_config)
        set_input_tensor_shapes_to_3d(model_config)

    if is_histo:
        model_config.n_channels = 4
        if hasattr(model_config, "architecture"):
            model_config.architecture["out_channels"] = 4

    csv_dataframe = create_csv_modality_labeling_type_path(
        GENERAL_DATA_DIR, modality, labeling_type
    )

    with ContextManagerTests(
        test_dir=TEST_DIR,
        test_name=test_name,
        output_dir=OUTPUT_DIR,
        inference_output_dir=INFERENCE_OUTPUT_DIR,
    ):
        example_dataframe = pd.read_csv(csv_dataframe)
        training_manager = TrainingManager(
            train_dataframe=example_dataframe,
            output_dir=OUTPUT_DIR,
            global_config=global_config,
            model_config=model_config,
            resume=False,
            reset=False,
        )
        training_manager.run_training()

        inference_kwargs = {
            "model_config": model_config,
            "global_config": global_config,
            "model_dir": OUTPUT_DIR,
            "output_dir": INFERENCE_OUTPUT_DIR,
        }
        if "vqvae" in config_path:
            inference_kwargs["dataframe_reconstruction"] = example_dataframe

        inference_manager = InferenceManager(**inference_kwargs)
        inference_manager.run_inference()


@pytest.mark.parametrize(
    "config_name, modality, n_dimensions, is_histo",
    [
        ("dcgan", "2d_rad", 2, False),
        ("dcgan", "3d_rad", 3, False),
        ("dcgan", "2d_histo", 2, True),
        ("vqvae", "2d_rad", 2, False),
        ("vqvae", "3d_rad", 3, False),
        ("vqvae", "2d_histo", 2, True),
        ("ddpm", "2d_rad", 2, False),
        ("ddpm", "3d_rad", 3, False),
        ("ddpm", "2d_histo", 2, True),
    ],
)
def test_module(config_name, modality, n_dimensions, is_histo):
    config_path = os.path.join(TEST_DIR, f"../configs/module_config_{config_name}.yaml")
    for labeling_type in LABELING_TYPES:
        run_test(config_path, modality, n_dimensions, labeling_type, is_histo)


def test_module_config_pairs():
    from gandlf_synth.models.modules.module_factory import ModuleFactory

    available_modules = list(ModuleFactory.AVAILABE_MODULES.keys())
    available_configs = list(ModuleFactory.AVAILABE_MODULES.keys())

    for module in available_modules:
        assert (
            module in available_configs
        ), f"Module {module} does not have a corresponding config"
    for config in available_configs:
        assert (
            config in available_modules
        ), f"Config {config} does not have a corresponding module"
