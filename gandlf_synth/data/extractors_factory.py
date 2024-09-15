from gandlf_synth.data.csv_data_extractor import (
    CSVDataExtractor,
    CustomLabeledDataExtractor,
    UnlabeledDataExtractor,
    PatientLabeledDataExtractor,
)

from typing import Literal, Type


class DataExtractorFactory:
    """Factory class to create data extractors based on the labeling paradigm."""

    DATA_EXTRACTOR_OBJECTS = {
        "unlabeled": UnlabeledDataExtractor,
        "patient": PatientLabeledDataExtractor,
        "custom": CustomLabeledDataExtractor,
    }

    def get_data_extractor(
        self,
        labeling_paradigm: Literal["unlabeled", "patient", "custom"],
        dataset_path: str,
        channel_id: str,
    ) -> Type[CSVDataExtractor]:
        """
        Factory function to create a data extractor based on the labeling paradigm.

        Args:
            labeling_paradigm (str): Labeling paradigm to be used.
            dataset_path (str): Path to the dataset.
            channel_id (str): Channel ID to be used.

        Returns:
            CSVDataExtractor: A data extractor object based on the labeling paradigm.
        """

        assert labeling_paradigm in self.DATA_EXTRACTOR_OBJECTS.keys(), (
            f"Labeling paradigm {labeling_paradigm} not found. "
            f"Available paradigms: {self.DATA_EXTRACTOR_OBJECTS.keys()}"
        )
        return self.DATA_EXTRACTOR_OBJECTS[labeling_paradigm](dataset_path, channel_id)
