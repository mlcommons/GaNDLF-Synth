from typing import Optional

import pandas as pd

from gandlf_synth.training_manager import TrainingManager
from gandlf_synth.config_manager import ConfigManager
from gandlf_synth.inference_manager import InferenceManager


def main_run(
    config_path: str,
    output_dir: str,
    main_data_csv_path: Optional[str] = None,
    training: bool = False,
    resume: bool = True,
    reset: bool = False,
    val_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    inference_output_dir: Optional[str] = None,
    custom_checkpoint_path: Optional[str] = None,
):
    """
    Main function to execute training or inference.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Path to the output directory, where data is saved druing training.
    This directory is also used in inference mode as the source for loading the model files.
        main_data_csv_path (str): Path to the main data CSV file. When in
    training mode, this argument is required as it will be used for training. When in inference,
    this file will be used for inference (utilized by models performing reconstruction).
        inference_output_dir (str): Path to the output directory used during inference, where the
    results of generation/reconstruction will be saved. Defaults to None, must be specified for inference.
        training (bool): Flag to indicate whether to run in training mode.
    Defaults to False.
        resume (bool): Flag to indicate whether to resume training from a
    checkpoint. Defaults to True.
        reset (bool): Flag to indicate whether to reset the output directory.
    Defaults to False.

        val_csv_path (str): Path to the validation data CSV file. Defaults to
    None.
        test_csv_path (str): Path to the test data CSV file. Defaults to None.
        val_ratio (float): Ratio of the validation data to use for training. If
    specified along with val_csv_path, the data from val_csv_path will be used.
    Defaults to None.
        test_ratio (float): Ratio of the test data to use for training. If
    specified along with test_csv_path, the data from test_csv_path will be used.
    Defaults to None.
        custom_checkpoint_path (str): Custom path to load the specific checkpoint either for
    training or inference. During training, it takes action only when `resume` is set to True.
    """

    config_manager = ConfigManager(config_path=config_path)
    global_config, model_config = config_manager.prepare_configs()

    main_input_dataframe = (
        pd.read_csv(main_data_csv_path) if main_data_csv_path is not None else None
    )

    if training:
        assert (
            main_input_dataframe is not None
        ), "When in training mode, `main_data_csv_path` must be specified!"
        val_dataframe = None
        if val_csv_path is not None:
            val_dataframe = pd.read_csv(val_csv_path)
        test_dataframe = None
        if test_csv_path is not None:
            test_dataframe = pd.read_csv(test_csv_path)
        # Reseting and resuming is handled by managers, so we do not validate it here.
        training_manager = TrainingManager(
            train_dataframe=main_input_dataframe,
            output_dir=output_dir,
            global_config=global_config,
            model_config=model_config,
            reset=reset,
            resume=resume,
            val_dataframe=val_dataframe,
            test_dataframe=test_dataframe,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            custom_checkpoint_path=custom_checkpoint_path,
        )
        training_manager.run_training()

    if not training:
        assert (
            inference_output_dir is not None
        ), "`inference_output_dir` must be specified when running in inference mode!"
        inference_manager = InferenceManager(
            global_config=global_config,
            model_config=model_config,
            model_dir=output_dir,
            output_dir=inference_output_dir,
            dataframe_reconstruction=main_input_dataframe,
            custom_checkpoint_path=custom_checkpoint_path,
        )
        inference_manager.run_inference()
