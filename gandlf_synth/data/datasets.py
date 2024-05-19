from abc import abstractmethod
from typing import Optional
import pandas as pd

import torchio as tio
from torchio.transforms import Compose
from torch.utils.data import Dataset, DataLoader


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

        channel_file_paths = [
            self.csv_data.loc[index, channel_column]
            for channel_column in channel_columns
        ]
        tio_scalar_image = tio.ScalarImage(channel_file_paths)

        if self.transforms:
            tio_scalar_image = self.transforms(tio_scalar_image)
        # TODO think if this is valid
        image = tio_scalar_image.data.squeeze(-1)  # if 2D the last dim is 1
        return image


class DatasetFactory:
    DATASET_OBJECTS = {"unlabeled": UnlabeledSynthesisDataset}

    def get_dataset(
        self,
        csv_path: str,
        transforms: Optional[Compose],
        labeling_paradigm: Optional[str] = "unlabeled",
    ) -> SynthesisDataset:
        """
        Factory function to create a dataset based on the labeling paradigm.

        Args:
            csv_path (str): Path to the CSV file containing the dataset information.
            transforms (Compose): Compose object containing the transforms to be applied.
            labeling_paradigm (str): Labeling paradigm to be used. Defaults to None.

        Returns:
            SynthesisDataset: A dataset object based on the labeling paradigm.
        """
        assert labeling_paradigm in self.DATASET_OBJECTS.keys(), (
            f"Labeling paradigm {labeling_paradigm} not found. "
            f"Available paradigms: {self.DATASET_OBJECTS.keys()}"
        )
        return self.DATASET_OBJECTS[labeling_paradigm](csv_path, transforms)


if __name__ == "__main__":
    # some basic testing, debugging purposes
    from GANDLF.data.preprocessing import global_preprocessing_dict

    example_transforms = Compose([global_preprocessing_dict["rescale"]()])
    dataset_factory = DatasetFactory()
    unlabeled_dataset = dataset_factory.get_dataset(
        csv_path="data/synth_data.csv", transforms=example_transforms
    )
    image = unlabeled_dataset[0]

    dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch[0].shape)
        break
