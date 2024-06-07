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

    def __init__(self, csv_path: str, transforms: Optional[Compose] = None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.transforms = transforms
        self.csv_data = pd.read_csv(csv_path)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.csv_data)


class UnlabeledSynthesisDataset(SynthesisDataset):
    """
    Unlabeled synthesis dataset.

    """

    # TODO We need to also think about how to handle the case if one
    # of the channels is a label map as we want to avoid applying the intensity
    # transforms to it. Maybe something similar to the torchio's `tio.Label` class
    # as in GaNDLF.
    def __getitem__(self, index):
        # check if there are multible channel columns (named Channel_n)
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
        image = tio_scalar_image.data.squeeze(-1).float()  # if 2D the last dim is 1
        return image
