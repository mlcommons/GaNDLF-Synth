import os

import pytest
from click.testing import CliRunner

from gandlf_synth.entrypoints.construct_csv import construct_csv

from . import CliCase, run_test_case, TmpDire, TmpFile, TmpNoEx

# This function is a place where a real logic is executed.
# For tests, we replace it with mock up, and check if this function is called
# with proper args for different cli commands
MOCK_PATH = "gandlf_synth.entrypoints.construct_csv._construct_csv"

# these files would be either created temporarily for test execution,
# or we ensure they do not exist
test_file_system = [
    TmpDire("input/"),
    TmpFile("channels_str.yaml", content="channels: _yaml1.gz,_yaml2.gz"),
    TmpFile("channels_list.yaml", content="channels:\n  - _yaml1.gz\n  - _yaml2.gz"),
    TmpFile(
        "channels_labels.yaml", content="channels: _yaml1.gz,_yaml2.gz\nlabel: _yaml.gz"
    ),
    TmpFile("output.csv", content="foobar"),
    TmpNoEx("output_na.csv"),
    TmpDire("output/"),
    TmpNoEx("path_na"),
]
test_cases = [
    CliCase(
        should_succeed=True,
        command_lines=[
            "--input-dir input --channels-id _t1.nii.gz,_t2.nii.gz --labeling-paradigm unlabeled --output-file output.csv",
            "-i input -ch _t1.nii.gz,_t2.nii.gz -l unlabeled -o output.csv",
        ],
        expected_args={
            "input_dir": os.path.normpath("input/"),
            "channels_id": "_t1.nii.gz,_t2.nii.gz",
            "labeling_paradigm": "unlabeled",
            "output_file": os.path.normpath("output.csv"),
        },
    ),
    CliCase(
        should_succeed=True,
        command_lines=[
            "--input-dir input --channels-id _t1.nii.gz,_t2.nii.gz --labeling-paradigm patient --output-file output.csv",
            "-i input -ch _t1.nii.gz,_t2.nii.gz -l patient -o output.csv",
        ],
        expected_args={
            "input_dir": os.path.normpath("input/"),
            "channels_id": "_t1.nii.gz,_t2.nii.gz",
            "labeling_paradigm": "patient",
            "output_file": os.path.normpath("output.csv"),
        },
    ),
    CliCase(
        should_succeed=True,
        command_lines=[
            "--input-dir input --channels-id _t1.nii.gz,_t2.nii.gz --labeling-paradigm custom --output-file output.csv",
            "-i input -ch _t1.nii.gz,_t2.nii.gz -l custom -o output.csv",
        ],
        expected_args={
            "input_dir": os.path.normpath("input/"),
            "channels_id": "_t1.nii.gz,_t2.nii.gz",
            "labeling_paradigm": "custom",
            "output_file": os.path.normpath("output.csv"),
        },
    ),
    CliCase(
        should_succeed=False,
        command_lines=[
            "--input-dir input --channels-id _t1.nii.gz,_t2.nii.gz --labeling-paradigm wrong_paradigm --output-file output.csv",
            "-i input -ch _t1.nii.gz,_t2.nii.gz -l wrong_paradigm -o output.csv",
        ],
    ),
]


@pytest.mark.parametrize("case", test_cases)
def test_case(cli_runner: CliRunner, case: CliCase):
    run_test_case(
        case=case,
        cli_runner=cli_runner,
        file_system_config=test_file_system,
        real_code_function_path=MOCK_PATH,
        cli_command=construct_csv,
        patched_return_value=None,
    )
