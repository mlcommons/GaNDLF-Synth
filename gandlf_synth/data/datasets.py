from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from torchio.transforms import Compose
import SimpleITK as sitk
import torchio as tio
import numpy as np


# Should we just inherit torch dataset? We have ou
class SynthesisDataset(ABC):
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


class UnlabeledSynthesisDataset(SynthesisDataset):
    """
    Unlabeled synthesis dataset.

    """

    # TODO this numpy array conversion is not necessary, just wanted to
    # make sure that the code runs. Here it is still WIP, still need to apply
    # the transforms. We need to also think about how to handle the case if one
    # of the channels is a label map as we want to avoid applying the intensity
    # transforms to it. Maybe something similar to the torchio's `tio.Label` class
    # as in GaNDLF.
    def __getitem__(self, index):
        # check if there are multible channel columns (named Channel_n)
        channel_columns = [col for col in self.csv_data.columns if "Channel_" in col]
        assert len(channel_columns) > 0, "No channel columns found in CSV."
        if len(channel_columns) > 1:
            # if there are multiple channel columns, we need to stack them
            channels = [
                sitk.ReadImage(self.csv_data.loc[index, channel_column])
                for channel_column in channel_columns
            ]
            # we obtain the arrays of shape (num_channels, height, width, depth) depth
            # is optional in 3D images
            channels_array = np.stack(
                [sitk.GetArrayFromImage(channel) for channel in channels], axis=0
            )
            return channels_array
        channel_columns = channel_columns[0]
        channels = sitk.ReadImage(self.csv_data.loc[index, channel_columns])
        channels_array = sitk.GetArrayFromImage(channels)
        return channels_array


if __name__ == "__main__":
    unlabeled_dataset = UnlabeledSynthesisDataset("output.csv")
    image = unlabeled_dataset[0]
    print(image.shape)
