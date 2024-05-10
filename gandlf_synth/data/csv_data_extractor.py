import os
from abc import ABC, abstractmethod

import pandas as pd
from pathlib import Path


class CSVDataExtractor(ABC):
    """
    The base abstract class for extracting data from a CSV file.
    The subsequent classes will implement the extract_data method to extract data from the CSV file
    based on the labeling mode specified in the config. The extracted data will be stored in the
    standard GaNDLF data format in a CSV.
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)

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
