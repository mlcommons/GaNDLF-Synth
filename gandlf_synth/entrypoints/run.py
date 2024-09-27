#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click


from gandlf_synth.entrypoints import append_copyright_to_help
from gandlf_synth.version import __version__
from gandlf_synth.cli.main_run import main_run


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    help="Path to the configuration file containing all parameters.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--main-data-csv-path",
    "-dt",
    required=False,
    help="Path to the CSV file which contains either the training data or the data to be used for inference.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--model-dir",
    "-m-dir",
    required=True,
    help="Path to the output directory where the results will be saved.",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--training/--inference",
    "-t/-i",
    required=True,
    help="Flag to indicate whether to run in training or inference mode. For inference, there needs to be a valid checkpoint in the --model-dir.",
)
@click.option(
    "--resume",
    "-rs",
    is_flag=True,
    help="Resume previous training by only keeping model dict in 'model-dir'",
)
@click.option(
    "--reset",
    "-rt",
    is_flag=True,
    help="Completely resets the previous run by deleting 'model-dir'",
)
@click.option(
    "--val-csv-path",
    "-v_csv",
    required=False,
    type=str,
    help="Optional path to the CSV file which contains the validation data used during training.",
)
@click.option(
    "--test-csv-path",
    "-t_csv",
    required=False,
    type=str,
    help="Optional path to the CSV file which contains the test data used after training.",
)
@click.option(
    "--val-ratio",
    "-v_r",
    required=False,
    default=0.0,
    type=float,
    help="Optional ratio of the validation data to use for training. If specified along with val_csv_path, the data from val_csv_path will be used.",
)
@click.option(
    "--test-ratio",
    "-t_r",
    required=False,
    default=0.0,
    type=float,
    help="Optional ratio of the test data to use for training. If specified along with test_csv_path, the data from test_csv_path will be used.",
)
@click.option(
    "--inference-output-dir",
    "-i-dir",
    required=False,
    type=str,
    help="Optional path to the CSV file which contains the validation data used during training.",
)
@click.option(
    "--custom-checkpoint-path",
    "-ckpt-path",
    required=False,
    type=str,
    help="Optional path to specify from which checkpoint to resume training or to use for inference. In training, it takes action only if --resume is set.",
)
@append_copyright_to_help
def run(
    config: str,
    main_data_csv_path: str,
    model_dir: str,
    training: bool,
    resume: bool,
    reset: bool,
    val_csv_path: str,
    test_csv_path: str,
    val_ratio: float,
    test_ratio: float,
    inference_output_dir: str,
    custom_checkpoint_path: str,
):
    main_run(
        config_path=config,
        main_data_csv_path=main_data_csv_path,
        output_dir=model_dir,
        training=training,
        resume=resume,
        reset=reset,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        inference_output_dir=inference_output_dir,
        custom_checkpoint_path=custom_checkpoint_path,
    )


if __name__ == "__main__":
    run()
