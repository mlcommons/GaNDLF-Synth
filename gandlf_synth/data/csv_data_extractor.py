import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd


def extend_filenames_to_absolute_paths(filenames: List[str]) -> List[str]:
    """
    Automatically find the absolute path of the files.

    Args:
        filenames (List[str]): The list of filenames.
    """
    return [os.path.abspath(filename) for filename in filenames]


class CSVDataExtractor(ABC):
    """
    The base abstract class for extracting data from a CSV file.
    The subsequent classes will implement the extract_data method to extract data from the CSV file
    based on the labeling mode specified in the config. The extracted data will be stored in the
    standard GaNDLF data format in a CSV.
    """

    def __init__(self, dataset_path: str, channel_id: str) -> None:
        """
        Initialize the CSVDataExtractor object.

        Args:
            dataset_path (str): The path to the dataset.
            channel_id (str): The channel ID to identify the sample (e.g. slice.nii.gz).
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.channel_id = channel_id
        self.channel_id_list = channel_id.split(",")

    @abstractmethod
    def _extract_data(self, dataset_path: Path) -> pd.DataFrame:
        pass

    @staticmethod
    def _save_csv(dataframe: pd.DataFrame, output_path: str) -> None:
        """
        Save the dataframe to a CSV file.

        Args:
            dataframe (pd.DataFrame): The dataframe to be saved.
            output_path (str): The path to the CSV file.
        """
        dataframe.to_csv(output_path, index=False)

    def extract_csv_data(self, output_path: str) -> None:
        """
        Extract data from the CSV file and save it to a new CSV file.

        Args:
            output_path (str): The path to the output CSV file.
        """
        data = self._extract_data(self.dataset_path)
        self._save_csv(data, output_path)


class UnlabeledDataExtractor(CSVDataExtractor):
    """
    Extractor for the case when the user operates in unconditional mode.
    Reads all image files from a given directory and stores them in a CSV file
    in GaNDLF standard format. As there is possibility of multiple channels,
    the following directory structure is assumed:
    dataset
    ├── subject1
    │   ├── t2w.nii.gz
    │   ├── t1.nii.gz
    ├── subject2
    │   ├── t2w.nii.gz
    │   ├── t1.nii.gz
    etc.
    Additional channels are not necessary. If the user provides multiple channel IDs,
    the extractor will demand that all channels are present in each subject directory.
    """

    def _extract_data(self, dataset_path: Path) -> pd.DataFrame:
        """
        Extract data from the CSV file.

        Args:
            dataset_path (Path): The path to the dataset.

        Returns:
            pd.DataFrame: The extracted data.
        """
        dataframe = pd.DataFrame(
            columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
        )

        # find all image files in the dataset based on the channel IDs
        # TODO how we decide if we are saving the absolute path or relative path?
        # for now I am saving the absolute path
        rows = []
        for dirpath, _, files in os.walk(dataset_path):
            if files:
                channels_row_relative = [
                    os.path.join(dirpath, file)
                    for file in files
                    if any(channel_id in file for channel_id in self.channel_id_list)
                ]
                channels_row_abs = extend_filenames_to_absolute_paths(
                    channels_row_relative
                )
                assert len(channels_row_abs) == len(self.channel_id_list), (
                    f"Missing channels in {dirpath}. "
                    f"Expected channels: {self.channel_id_list}. "
                    f"Found channels: {files}"
                )
                rows.append(channels_row_abs)
        dataframe = pd.DataFrame(
            rows, columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
        )
        return dataframe


class CustomLabeledDataExtractor(CSVDataExtractor):
    """
    Extractor for the case when the user operates in conditional mode.
    Reads all image files and creates labels for them. The labels are
    determined based on the demanded directory structure.
    Example directory structure:
    dataset
    ├── class1
    │   ├── subject1
    │   │   ├── t2w.nii.gz
    │   │   ├── t1.nii.gz
    ├── class2
    │   ├── subject2
    │   │   ├── t2w.nii.gz
    │   │   ├── t1.nii.gz
    etc.
    We require user to place images in directories named after the class.
    """

    def _extract_data(self, dataset_path: Path) -> pd.DataFrame:
        """
        Extract data from the CSV file.

        Args:
            dataset_path (Path): The path to the dataset.

        Returns:
            pd.DataFrame: The extracted data.
        """

        def determine_label_from_path(path: Union[Path, str]) -> str:
            """
            Determine the label from the path. Assumes that the label is
            the name of the parent directory of the parent directory.

            Args:
                path (Union[Path,str]): The path to the file.

            Returns:
                str: The label.
            """
            if isinstance(path, str):
                path = Path(path)
            return path.parent.name

        class_names = os.listdir(dataset_path)
        class_to_id_mapping = {
            class_name: i for i, class_name in enumerate(class_names)
        }
        rows = []
        for dirpath, _, files in os.walk(dataset_path):
            if files:
                channels_row_relative = [
                    os.path.join(dirpath, file)
                    for file in files
                    if any(channel_id in file for channel_id in self.channel_id_list)
                ]
                channels_row_abs = extend_filenames_to_absolute_paths(
                    channels_row_relative
                )
                assert len(channels_row_abs) == len(self.channel_id_list), (
                    f"Missing channels in {dirpath}. "
                    f"Expected channels: {self.channel_id_list}. "
                    f"Found channels: {files}"
                )
                label = determine_label_from_path(dirpath)
                label_id = class_to_id_mapping[label]
                channels_row_abs.extend([label_id, label])
                rows.append(channels_row_abs)
        dataframe = pd.DataFrame(
            rows,
            columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
            + ["Label"]
            + ["LabelMapping"],
        )
        return dataframe


class PatientLabeledDataExtractor(CSVDataExtractor):
    """
    Extractor for the case when the user operates in conditional mode.
    Reads all image files and creates labels for them. The labels are
    determined based on the patient ID.
    Example directory structure:

    dataset
    ├── patient1
    │   ├── t2w.nii.gz
    │   ├── t1.nii.gz
    ├── patient2
    │   ├── t2w.nii.gz
    │   ├── t1.nii.gz
    etc.

    """

    def _extract_data(self, dataset_path: Path) -> pd.DataFrame:
        """
        Extract data from the CSV file.

        Args:
            dataset_path (Path): The path to the dataset.

        Returns:
            pd.DataFrame: The extracted data.
        """
        dataframe = pd.DataFrame(
            columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
        )

        # find all image files in the dataset based on the channel IDs
        # TODO how we decide if we are saving the absolute path or relative path?
        # for now I am saving the absolute path
        rows = []
        class_names = os.listdir(dataset_path)
        class_to_id_mapping = {
            class_name: i for i, class_name in enumerate(class_names)
        }
        for dirpath, _, files in os.walk(dataset_path):
            if files:
                channels_row_relative = [
                    os.path.join(dirpath, file)
                    for file in files
                    if any(channel_id in file for channel_id in self.channel_id_list)
                ]
                channels_row_abs = extend_filenames_to_absolute_paths(
                    channels_row_relative
                )
                assert len(channels_row_abs) == len(self.channel_id_list), (
                    f"Missing channels in {dirpath}. "
                    f"Expected channels: {self.channel_id_list}. "
                    f"Found channels: {files}"
                )
                label = os.path.basename(dirpath)
                label_id = class_to_id_mapping[label]
                channels_row_abs.extend([label_id, label])
                rows.append(channels_row_abs)

        dataframe = pd.DataFrame(
            rows,
            columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
            + ["Label"]
            + ["LabelMapping"],
        )
        return dataframe


if __name__ == "__main__":
    dataset_path = "testing/data/2d_rad/unlabeled"
    channel_id = "t2w.nii.gz,t1.nii.gz"
    output_path = "testing/patient_data_test.csv"
    extractor = PatientLabeledDataExtractor(dataset_path, channel_id)
    extractor.extract_csv_data(output_path)
