import torch
import pandas as pd
from torchio.transforms import Compose

from gandlf_synth.data.datasets import SynthesisDataset, UnlabeledSynthesisDataset
from gandlf_synth.utils.managers_utils import prepare_transforms

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


class InferenceDatasetFactory:
    def __init__(
        self,
        global_config: dict,
        model_config: dict,
        dataframe_reconstruction: Optional[pd.DataFrame],
    ):
        self.global_config = global_config
        self.dataframe_reconstruction = dataframe_reconstruction
        self.model_config = model_config

    def _unlabeled_inference_dataset(self):
        n_images_to_generate = self.global_config["inference_parameters"][
            "n_images_to_generate"
        ]
        indices_list = list(range(n_images_to_generate))
        dataset = torch.utils.data.TensorDataset(torch.tensor(indices_list))
        return dataset

    def _labeled_inference_dataset(self):
        per_class_n_images_to_generate = self.global_config["inference_parameters"][
            "n_images_to_generate"
        ]
        indices_list = []
        labels_list = []
        for class_label, n_images_to_generate in per_class_n_images_to_generate.items():
            indices_list.extend(list(range(n_images_to_generate)))
            labels_list.extend([class_label] * n_images_to_generate)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(indices_list), torch.tensor(labels_list)
        )
        return dataset

    def _reconstruction_inference_dataset(self):
        """
        Prepare the dataloader for the inference process if reconstruction
        data is provided.

        Returns:
            torch.utils.data.DataLoader: The dataloader for the inference process.
        """
        transforms = prepare_transforms(
            augmentations_config=self.global_config.get("data_augmentations"),
            preprocessing_config=self.global_config.get("data_preprocessing"),
            mode="inference",
            input_shape=self.model_config.tensor_shape,
        )
        dataset_factory = DatasetFactory()
        dataset = dataset_factory.get_dataset(
            self.dataframe_reconstruction,
            transforms,
            labeling_paradigm=self.model_config.labeling_paradigm,
        )
        return dataset

    # TODO: Think if this is the way to do it.
    def get_inference_dataset(self) -> torch.utils.data.Dataset:
        labeling_paradigm = self.model_config.labeling_paradigm
        module_type = (
            "reconstruction"
            if self.dataframe_reconstruction is not None
            else "generation"
        )
        if module_type == "reconstruction":
            return self._reconstruction_inference_dataset()
        elif labeling_paradigm == "unlabeled" and module_type == "generation":
            return self._unlabeled_inference_dataset()
        elif labeling_paradigm == "labeled" and module_type == "generation":
            return self._labeled_inference_dataset()
