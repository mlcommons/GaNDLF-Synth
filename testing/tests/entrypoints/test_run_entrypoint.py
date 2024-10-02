import os
import pytest
from click.testing import CliRunner

from gandlf_synth.entrypoints.run import run
from . import CliCase, run_test_case, TmpDire, TmpFile

# Mock path for the main_run function
MOCK_PATH = "gandlf_synth.entrypoints.run.main_run"

# Temporary file system setup
test_file_system = [
    TmpFile("config.yaml", content="config content"),
    TmpFile("main_data.csv", content="main data"),
    TmpFile("val_data.csv", content="validation data"),
    TmpFile("test_data.csv", content="test data"),
    TmpDire("model_dir/"),
    TmpDire("inference_output/"),
    TmpFile("custom_checkpoint.pth", content="checkpoint data"),
]

test_cases = [
    # Basic training case
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --training",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -t",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": False,
            "reset": False,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
    # Basic inference case
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --inference-output-dir inference_output",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -i-dir inference_output",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": False,
            "resume": False,
            "reset": False,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": "inference_output",
            "custom_checkpoint_path": None,
        },
    ),
    # test-val from dataframe during training
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --training --val-csv-path val_data.csv --test-csv-path test_data.csv",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -t  -v-csv val_data.csv -t-csv test_data.csv",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": False,
            "reset": False,
            "val_csv_path": "val_data.csv",
            "test_csv_path": "test_data.csv",
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
    # test-val from ratio during training
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --training --val-ratio 0.2 --test-ratio 0.1",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -t -vr 0.2 -tr 0.1",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": False,
            "reset": False,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
    # test-val both from ratio and dataframe during training
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --training --val-ratio 0.2 --test-ratio 0.1 --val-csv-path val_data.csv --test-csv-path test_data.csv",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -t -vr 0.2 -tr 0.1 -v-csv val_data.csv -t-csv test_data.csv",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": False,
            "reset": False,
            "val_csv_path": "val_data.csv",
            "test_csv_path": "test_data.csv",
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
    # inference with custom checkpoint
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir  --custom-checkpoint-path custom_checkpoint.pth --inference-output-dir inference_output",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -ckpt-path custom_checkpoint.pth -i-dir inference_output",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": False,
            "resume": False,
            "reset": False,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": "inference_output",
            "custom_checkpoint_path": "custom_checkpoint.pth",
        },
    ),
    # resume training with custom checkpoint
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --resume --training --custom-checkpoint-path custom_checkpoint.pth",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -rs -t -ckpt-path custom_checkpoint.pth",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": True,
            "reset": False,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": None,
            "custom_checkpoint_path": "custom_checkpoint.pth",
        },
    ),
    # reset training
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --reset --training",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -rt -t",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": False,
            "reset": True,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
    # both reset and resume training - resume takes precedence
    CliCase(
        should_succeed=True,
        command_lines=[
            "--config config.yaml --main-data-csv-path main_data.csv --model-dir model_dir --reset --resume --training",
            "-c config.yaml -dt main_data.csv -m-dir model_dir -rs -rt -t",
        ],
        expected_args={
            "config_path": "config.yaml",
            "main_data_csv_path": "main_data.csv",
            "output_dir": os.path.normpath("model_dir"),
            "training": True,
            "resume": True,
            "reset": True,
            "val_csv_path": None,
            "test_csv_path": None,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "inference_output_dir": None,
            "custom_checkpoint_path": None,
        },
    ),
]


@pytest.mark.parametrize("case", test_cases)
def test_case_run(cli_runner: CliRunner, case: CliCase):
    run_test_case(
        case=case,
        cli_runner=cli_runner,
        file_system_config=test_file_system,
        real_code_function_path=MOCK_PATH,
        cli_command=run,
        patched_return_value=None,
    )
