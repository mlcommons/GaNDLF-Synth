from torchio.transforms import Compose

from gandlf_synth.data.datasets import SynthesisDataset, UnlabeledSynthesisDataset

from typing import Optional


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
