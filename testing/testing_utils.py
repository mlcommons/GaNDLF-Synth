import os
import shutil
import zipfile
import gdown
import subprocess
from datetime import datetime
from gandlf_synth.models.modules.module_abc import SynthesisModule

from typing import List, Optional, Type

UNIT_TEST_DATA_SOURCE = (
    "https://drive.google.com/uc?id=12utErBXZiO_0hspmzUlAQKlN9u-manH_"
)


class ContextManagerTests:
    """
    Context manager ensuring that certain operations are performed before and after the tests.
    It is used to clean the output directories before and after the tests.
    If the tests fail, the output directories are copied to a failed_runs directory for inspection.
    """

    def __init__(
        self,
        test_dir: str,
        test_name: str,
        output_dir: str,
        inference_output_dir: Optional[str] = None,
    ):
        """
        Initialize the context manager.

        Args:
            test_dir (str): The directory where the tests are run.
            test_name (str): The name of the test.
            output_dir (str): The output directory where the results are stored.
            inference_output_dir (str): The output directory where the inference results are stored.
        """
        self.test_dir = test_dir
        self.test_name = test_name
        self.output_dir = output_dir
        self.inference_output_dir = inference_output_dir

    def __enter__(self):
        """
        Method to be executed before the tests.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method to be executed after the tests.
        """
        if exc_type is not None and exc_type is not KeyboardInterrupt:
            failed_runs_dir = os.path.join(self.test_dir, "output_failed")
            if not os.path.exists(failed_runs_dir):
                os.mkdir(failed_runs_dir)
            if os.path.exists(self.output_dir):
                shutil.copytree(
                    self.output_dir,
                    os.path.join(
                        failed_runs_dir,
                        f"output_failed_{self.test_name}_date_{datetime.now()}",
                    ),
                )
            if os.path.exists(self.inference_output_dir):
                shutil.copytree(
                    self.inference_output_dir,
                    os.path.join(
                        failed_runs_dir,
                        f"inference_output_failed_{self.test_name}_date_{datetime.now()}",
                    ),
                )
        with os.scandir(self.output_dir) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
        if self.inference_output_dir is not None:
            with os.scandir(self.inference_output_dir) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)


def parse_available_module(module_name: str) -> List[str]:
    """
    Helper method to parse the module name into its components (labeling paradigm and model name).
    Used to replace the model name and labeling paradigm in the config file to check all
    available modules and configs in one go.

    Args:
        module_name (str): The module name from available modules.

    Returns:
        List[str]: The list containing the labeling paradigm and model name.

    """
    return module_name.split("_")


def prerequisites_hook_download_data():
    """
    Utility function to download the data for the unit tests if the data folder
    does not exist. The data is downloaded from the Google Drive link.
    """

    print("Downloading data...")
    test_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(test_dir_path, "data")
    zipfile_path = os.path.join(test_dir_path, "data.zip")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        gdown.download(UNIT_TEST_DATA_SOURCE, zipfile_path, quiet=False)
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(test_dir_path)
        os.remove(zipfile_path)
        print("Data downloaded successfully.")
    else:
        print("Data already exists.")


def construct_csv_files():
    """
    Utility function to construct the csv files for the data.
    The data is assumed to be in the following structure:

    gandlf-synth/testing/data
    ├── 2d_histo
    │   ├── labeled
    │   └── unlabeled
    ├── 2d_rad
    │   ├── labeled
    │   └── unlabeled
    └── 3d_rad
        ├── labeled
        └── unlabeled

    """
    filenames_map = {"unlabeled": "unlabeled_data.csv", "labeled": "labeled_data.csv"}
    test_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(test_dir_path, "data")
    data_modalities_paths = [
        os.path.join(data_path, modality) for modality in os.listdir(data_path)
    ]
    for data_modality_path in data_modalities_paths:
        modality_subdirs_paths = [
            os.path.join(data_modality_path, subdir)
            for subdir in os.listdir(data_modality_path)
            if os.path.isdir(os.path.join(data_modality_path, subdir))
        ]
        modality_dimension = os.path.basename(
            data_modality_path
        )  # modality combinded with dimension info
        modality = modality_dimension.split("_")[1]
        channel_id = "t1.nii.gz,t2w.nii.gz" if modality == "rad" else ".tiff"
        for modality_subdir_path in modality_subdirs_paths:
            labeling_type = os.path.basename(modality_subdir_path)
            csv_file_path = os.path.join(
                data_modality_path,
                f"{modality_dimension}_{filenames_map[labeling_type]}",
            )
            subprocess.run(
                [
                    "gandlf",
                    "construct-csv",
                    "--input-dir",
                    modality_subdir_path,
                    "--output-file",
                    csv_file_path,
                    "--channels-id",
                    channel_id,
                ]
            )


def set_3d_dataloader_resize(global_config: dict):
    """
    Utility function to set the config parameters for resizing the 3D data for all dataloaders.
    Done inplace.
    Args:
        global_config (dict): The global configuration dictionary.

    """
    size = 64
    global_config["data_preprocessing"]["test"]["resize"] = [size, size, size]
    global_config["data_preprocessing"]["train"]["resize"] = [size, size, size]
    global_config["data_preprocessing"]["val"]["resize"] = [size, size, size]
    global_config["data_preprocessing"]["inference"]["resize"] = [size, size, size]


def set_input_tensor_shapes_to_3d(model_config: Type[SynthesisModule]):
    """
    Utility function to set the input and output tensor shapes to 3D.
    Done inplace.

    Args:
        model_config (Type[SynthesisModule]): The model configuration.
    """

    model_config.tensor_shape = [64, 64, 64]
    if hasattr(model_config, "input_shape"):
        model_config.input_shape = [64, 64, 64]
    if hasattr(model_config, "output_shape"):
        model_config.output_shape = [64, 64, 64]


def create_csv_modality_labeling_type_path(
    main_data_dir: str, modality: str, labeling_type: str
) -> str:
    """
    Helper function to create the path to the csv file based on the modality and labeling type.

    Args:
        main_data_dir (str): The main data directory.
        modality (str): The modality of the data.
        labeling_type (str): The labeling type of the data.

    Returns:
        str: The path to the csv file.
    """
    csv_path = os.path.join(
        main_data_dir, modality, f"{modality}_{labeling_type}_data.csv"
    )
    return csv_path
