from abc import abstractmethod
from typing import Optional
import pandas as pd

import torchio as tio
from torchio.transforms import Compose
from torch.utils.data import Dataset


# Can we just inherit from the torch dataset? Or do we need to define
# out own abstract class?
class SynthesisDataset(Dataset):
    """
    Base abstraction for a synthesis dataset.
    """

    def __init__(
        self, input_dataframe: pd.DataFrame, transforms: Optional[Compose] = None
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.csv_data = input_dataframe

    # TODO We need to also think about how to handle the case if one
    # of the channels is a label map as we want to avoid applying the intensity
    # transforms to it. Maybe something similar to the torchio's `tio.Label` class
    # as in GaNDLF.
    def _prepare_multichannel_image(self, index):
        """
        General class for reading one row of the CSV file and returning a multichannel image
        in torchio format. Applies the transforms if they are provided.

        Args:
            index (int): Index of the row in the CSV file.

        Returns:
            torch.Tensor: Multichannel image in torchio format.
        """
        channel_columns = [col for col in self.csv_data.columns if "Channel_" in col]
        assert len(channel_columns) > 0, "No channel columns found in CSV."

        channel_file_paths = [
            self.csv_data.loc[index, channel_column]
            for channel_column in channel_columns
        ]
        tio_scalar_image = tio.ScalarImage(channel_file_paths)

        if self.transforms:
            tio_scalar_image = self.transforms(tio_scalar_image)
        # TODO think if this is valid
        image = tio_scalar_image.data.squeeze(-1).float()
        return image

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.csv_data)


class UnlabeledSynthesisDataset(SynthesisDataset):
    """
    Unlabeled synthesis dataset.
    """

    def __getitem__(self, index):
        return self._prepare_multichannel_image(index)


class LabeledSynthesisDataset(SynthesisDataset):
    """
    Labelled synthesis dataset. Supports all types of single-label scenarios
    """

    def _process_label(self, index):
        """
        Get the label of the image at the given index from source CSV file.

        Args:
            index (int): Index of the row in the CSV file.

        Returns:
            torch.Tensor: Label of the image.
        """
        return self.csv_data.iloc[index]["Label"]

    def __getitem__(self, index):
        image = self._prepare_multichannel_image(index)
        label = self._process_label(index)
        return image, label
