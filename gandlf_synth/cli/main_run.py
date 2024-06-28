from typing import Optional

import pandas as pd

from gandlf_synth.training_manager import TrainingManager
from gandlf_synth.config_manager import ConfigManager


def main_run(
    config_path: str,
    main_data_csv_path: str,
    output_dir: str,
    training: bool = False,
    resume: bool = True,
    reset: bool = False,
    device: str = "cpu",
    val_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
):
    """
    Main function to execute training or inference.

    Args:
        config_path (str): Path to the configuration file.
        main_data_csv_path (str): Path to the main data CSV file. When in
    training mode, this file will be used for training. When in inference,
    this file will be used for inference.
        output_dir (str): Path to the output directory.
        training (bool): Flag to indicate whether to run in training mode.
    Defaults to False.
        resume (bool): Flag to indicate whether to resume training from a
    checkpoint. Defaults to True.
        reset (bool): Flag to indicate whether to reset the output directory.
    Defaults to False.
        device (str): Device to use for training or inference. Defaults to
    "cpu".
        val_csv_path (str): Path to the validation data CSV file. Defaults to
    None.
        test_csv_path (str): Path to the test data CSV file. Defaults to None.
        val_ratio (float): Ratio of the validation data to use for training. If
    specified along with val_csv_path, the data from val_csv_path will be used.
    Defaults to None.
        test_ratio (float): Ratio of the test data to use for training. If
    specified along with test_csv_path, the data from test_csv_path will be used.
    Defaults to None.
    """

    config_manager = ConfigManager(config_path=config_path)
    global_config, model_config = config_manager.prepare_configs()
    train_dataframe = pd.read_csv(main_data_csv_path)

    if training:
        val_dataframe = None
        test_dataframe = None
        if val_csv_path is not None:
            val_dataframe = pd.read_csv(val_csv_path)
        if test_csv_path is not None:
            test_dataframe = pd.read_csv(test_csv_path)
        # Reseting and resuming is handled by managers, so we do not validate it here.
        training_manager = TrainingManager(
            train_dataframe=train_dataframe,
            output_dir=output_dir,
            global_config=global_config,
            model_config=model_config,
            reset=reset,
            resume=resume,
            val_dataframe=val_dataframe,
            test_dataframe=test_dataframe,
            device=device,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        training_manager.run_training()

    if not training:
        print("Inference mode is not implemented yet. Exiting...")
        pass
