import pandas as pd
from torchio.transforms import Compose

from gandlf_synth.data.datasets import SynthesisDataset, UnlabeledSynthesisDataset

from typing import Optional


class DatasetFactory:
    """Class to create synthesis datasets based on the labeling paradigm."""

    DATASET_OBJECTS = {"unlabeled": UnlabeledSynthesisDataset}

    def get_dataset(
        self,
        dataframe: pd.DataFrame,
        transforms: Optional[Compose],
        labeling_paradigm: Optional[str] = "unlabeled",
    ) -> SynthesisDataset:
        """
        Factory function to create a dataset based on the labeling paradigm.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the data.
            transforms (Compose): Compose object containing the transforms to be applied.
            labeling_paradigm (str): Labeling paradigm to be used. Defaults to "unlabeled".

        Returns:
            SynthesisDataset: A dataset object based on the labeling paradigm.
        """
        assert labeling_paradigm in self.DATASET_OBJECTS.keys(), (
            f"Labeling paradigm {labeling_paradigm} not found. "
            f"Available paradigms: {self.DATASET_OBJECTS.keys()}"
        )
        return self.DATASET_OBJECTS[labeling_paradigm](dataframe, transforms)
