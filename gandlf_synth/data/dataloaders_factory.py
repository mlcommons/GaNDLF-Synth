from torch.utils.data import DataLoader
from gandlf_synth.data.datasets import SynthesisDataset
from typing import Type


class DataloaderFactory:
    """
    Class responsible for creating dataloaders.
    """

    def __init__(self, params: dict):
        """
        Initialize the DataloaderFactory.

        Args:
            params (dict): Dictionary containing the parameters.
        """

        self.params = params
        self.global_batch_size = self.params["batch_size"]

    def _get_dataloder(self, dataloader_params: dict, dataset: Type[SynthesisDataset]):
        """
        Get the dataloader given the configuration parameters.

        Args:
            dataloader_params (dict): The dataloader configuration parameters.
            dataset (SynthesisDataset): The dataset object.

        Returns:
            DataLoader: The dataloader object.
        """

        batch_size = self.global_batch_size
        if "batch_size" in dataloader_params:
            batch_size = dataloader_params["batch_size"]

        return DataLoader(**dataloader_params, batch_size=batch_size, dataset=dataset)

    def get_training_dataloader(self, dataset: Type[SynthesisDataset]) -> DataLoader:
        """
        Get the training dataloader.

        Args:
            dataset (SynthesisDataset): The dataset object.

        Returns:
            DataLoader: The training dataloader.
        """
        return self._get_dataloder(self.params["dataloader_config"]["train"], dataset)

    def get_validation_dataloader(self, dataset: Type[SynthesisDataset]) -> DataLoader:
        """
        Get the validation dataloader.

        Args:
            dataset (SynthesisDataset): The dataset object.

        Returns:
            DataLoader: The validation dataloader.
        """

        return self._get_dataloder(
            self.params["dataloader_config"]["validation"], dataset
        )

    def get_testing_dataloader(self, dataset: Type[SynthesisDataset]) -> DataLoader:
        """
        Get the testing dataloader.

        Args:
            dataset (SynthesisDataset): The dataset object.

        Returns:
            DataLoader: The testing dataloader.
        """

        return self._get_dataloder(self.params["dataloader_config"]["test"], dataset)

    def get_inference_dataloader(self, dataset: Type[SynthesisDataset]) -> DataLoader:
        """
        Get the inference dataloader.

        Args:
            dataset (SynthesisDataset): The dataset object.

        Returns:
            DataLoader: The inference dataloader.
        """
        return self._get_dataloder(
            self.params["dataloader_config"]["inference"], dataset
        )
