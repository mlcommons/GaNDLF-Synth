#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

from gandlf_synth.data.extractors_factory import DataExtractorFactory
from gandlf_synth.entrypoints import append_copyright_to_help
from gandlf_synth.version import __version__


@click.command()
@click.option(
    "--input_dir",
    "-i",
    required=True,
    help="Path to the input directory.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--channels_id",
    "-ch",
    required=True,
    help="Channel ID to be used. This is a comma-separated string, denoting the channel IDs, e.g., 't1w.nii.gz,t2w.nii.gz'.",
)
@click.option(
    "--labeling_paradigm",
    "-l",
    required=True,
    help="Labeling paradigm to be used. Available paradigms: ['unlabeled', 'patient', 'custom']",
)
@click.option(
    "--output_path",
    "-o",
    required=True,
    help="Path to the output CSV file.",
    type=click.Path(file_okay=True, dir_okay=False),
)
@append_copyright_to_help
def construct_csv(
    input_dir: str, channels_id: str, labeling_paradigm: str, output_path: str
):
    """
    Construct a CSV file based on the labeling paradigm.

    Args:
        input_dir (str): Path to the input directory.
        channels_id (str): Channel ID to be used. This is a comma-separated string,
    denoting the channel IDs, e.g., "t1w.nii.gz,t2w.nii.gz".
        labeling_paradigm (str): Labeling paradigm to be used. Available paradigms:
    ["unlabeled", "patient", "custom"]
        output_path (str): Path to the output CSV file.
    """
    extractor = DataExtractorFactory().get_data_extractor(
        labeling_paradigm, input_dir, channels_id
    )
    extractor.extract_csv_data(output_path)


if __name__ == "__main__":
    construct_csv()
