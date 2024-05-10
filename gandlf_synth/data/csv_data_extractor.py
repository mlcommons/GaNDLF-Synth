import os
from glob import glob
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


def append_row_to_dataframe(dataframe: pd.DataFrame, row: list):
    """
    Append a row to the dataframe inplace.

    Args:
        dataframe (pd.DataFrame): The dataframe to which the row will be appended.
        row (list): The row to be appended.
    """
    dataframe.loc[len(dataframe)] = row


def find_files_in_dir(directory: str, channel_id: str) -> List[str]:
    """
    Find all files in a directory that contain the channel ID. Sorts
    them based on the parent directory name.

    Args:
        directory (str): The path to the directory.
        channel_id (str): The channel ID.

    Returns:
        List[str]: A list of files that contain the channel ID.
    """

    channel_files = glob(f"{directory}/**/*{channel_id}", recursive=True)
    assert channel_files, f"No files found for channel {channel_id}"
    channel_files.sort(key=lambda x: os.path.basename(os.path.dirname(x)))
    return channel_files


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
    in GaNDLF standard format.

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
        # TODO this works for multiple channels, but assumes the same
        # number of files for each channel. Need to think if we really
        # need to support multiple channels.
        # files_for_channels = {
        #     channel_id: find_files_in_dir(dataset_path, channel_id)
        #     for channel_id in self.channel_id_list
        # }
        # rows = list((zip(*files_for_channels.values())))
        # rows = [list(row) for row in rows]
        # [append_row_to_dataframe(dataframe, row) for row in rows]

        # for now I assume we accept only one channel
        files = find_files_in_dir(dataset_path, self.channel_id)
        [append_row_to_dataframe(dataframe, [file]) for file in files]
        return dataframe


class LabeledDataExtractor(CSVDataExtractor):
    """
    Extractor for the case when the user operates in conditional mode.
    Reads all image files and creates labels for them. The labels are
    extracted from the names of directories containing the image files.
    Example directory structure:
    dataset
    ├── class1
    │   ├── image1.nii.gz
    │   ├── image2.nii.gz
        ├── class2
    │   ├── image3.nii.gz
    │   ├── image4.nii.gz
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
        dataframe = pd.DataFrame(
            columns=[f"Channel_{i}" for i in range(len(self.channel_id_list))]
            + ["Label"]
            + ["LabelMapping"]
        )
        files = find_files_in_dir(dataset_path, self.channel_id)
        unique_dirnames = list(set([Path(file).parent.name for file in files]))
        dirnames_to_labels = {dirname: i for i, dirname in enumerate(unique_dirnames)}
        [
            append_row_to_dataframe(
                dataframe,
                [
                    file,
                    dirnames_to_labels[Path(file).parent.name],
                    Path(file).parent.name,
                ],
            )
            for file in files
        ]
        return dataframe


if __name__ == "__main__":
    dataset_path = "testing/data/unlabeled"
    channel_id = ".nii.gz"
    output_path = "output.csv"
    extractor = UnlabeledDataExtractor(dataset_path, channel_id)
    extractor.extract_csv_data(output_path)

    dataset_path = "testing/data/labeled"
    channel_id = ".nii.gz"
    output_path = "output_labeled.csv"
    extractor = LabeledDataExtractor(dataset_path, channel_id)
    extractor.extract_csv_data(output_path)
