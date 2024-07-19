import os
import yaml
import shutil
from datetime import datetime

from typing import List


class ContextManagerTests:
    """
    Context manager ensuring that certain operations are performed before and after the tests.
    """

    def __init__(self, test_dir: str, test_name: str, output_dir: str):
        """
        Initialize the context manager.

        Args:
            test_dir (str): The directory where the tests are run.
            test_name (str): The name of the test.
            output_dir (str): The output directory where the results are stored.
        """
        self.test_dir = test_dir
        self.test_name = test_name
        self.output_dir = output_dir

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
        # Later we may move output dir sanitization here too, and other stuff
        # restore_config()
        with os.scandir(self.output_dir) as entries:
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)


def restore_config(config_path: str, original_config: dict):
    """
    Sanitizing function to restore the original config file after the tests are done, in
    case it is overwritten.

    Args:
        config_path (str): The path to the config file.
        original_config (dict): The original config dictionary.

    """

    with open(config_path, "w") as config_file:
        yaml.dump(original_config, config_file)


def parse_available_module(module_name: str) -> List[str]:
    """
    Helper method to parse the module name into its components (labeling paradigm and model name).
    Used to replace the model name and labeling paradigm in the config file to check all
    available modules and configs in one go.

    Args:
        module_name (str): The module name from available modules.

    Returns:
        labeling_paradigm (str): The labeling paradigm.
        model_name (str): The model name.

    """
    return module_name.split("_")


# TODO after the data will get uploaded to the cloud, we will replace the placeholder with the actual download
def prerequisites_hook_download_data():
    print("00: Downloading the sample data")
    print("Placeholder for downloading the data ran successfully")
