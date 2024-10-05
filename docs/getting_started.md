This document will help you get started with GaNDLF-Synth using a few representative examples.


## Installation

Follow the [installation instructions](./setup.md) to install GaNDLF-Synth. When the installation is complete, you should end up with the following shell, which indicates that the GaNDLF-Synth virtual environment has been activated:

```bash
(venv_gandlf) $> ### subsequent commands go here
```

<!-- 
Can we run it on codespaces?
## Running GaNDLF-Synth with GitHub Codespaces

Alternatively, you can launch a [Codespace](https://github.com/features/codespaces) for GaNDLF-Synth by clicking this link: 

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=master&repo=342691278)

A codespace will open in a web-based version of [Visual Studio Code](https://code.visualstudio.com/). The [dev container](.devcontainer/devcontainer.json) is fully configured with software needed for this project.

**Note**: Dev Containers is an open spec which is supported by [GitHub Codespaces](https://github.com/codespaces) and [other tools](https://containers.dev/supporting). -->


## Sample Data

Sample data will be used for our extensive automated unit tests in all examples. You can download the sample data from [this link](https://drive.google.com/uc?id=12utErBXZiO_0hspmzUlAQKlN9u-manH_). An example is shown below:

```bash
# continue from previous shell
(venv_gandlf) $> gdown https://drive.google.com/uc?id=12utErBXZiO_0hspmzUlAQKlN9u-manH_ -O ./gandlf_sample_data.zip
(venv_gandlf) $> unzip ./gandlf_sample_data.zip
# this should extract a directory called `data` in the current directory
```
The `data` directory content should look like the example below (for brevity, these locations shall be referred to as `${GANDLF_SYNTH_DATA}` in the rest of the document):

```bash
# continue from previous shell
(venv_gandlf) $>  ls data
2d_histo    2d_rad   3d_rad
# each of these directories contains data for a specific task in given labeling paradigm
```

**Note**: When using your own data, it is vital to correctly prepare the data. You can find the details on how to do it using GaNDLF core API [here](https://mlcommons.github.io/GaNDLF/usage#preparing-the-data).


## Train and use models

1. Download and extract the sample data as described in the [sample data](#sample-data). Alternatively, you can use your own data (see [constructing CSV in usage](./usage.md#constructing-the-data-csv) for an example).
2. [Construct the main data file](./usage.md#constructing-the-data-csv) that will be used for the entire computation cycle. For the sake of this document, we will use 3D radiology images in unlabeled mode, but the same steps can be followed for other modalities and labeling paradigms. For the sample data for this task, the base location is `${GANDLF_SYNTH_DATA}/3d_rad/unlabeled`, and it will be referred to as `${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}` in the rest of the document. The CSV should look like the example below:

    ```csv
    Channel_0,Channel_1
    ${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/003/t2w.nii.gz,${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/003/t1.nii.gz
    ${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/001/t2w.nii.gz,${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/001/t1.nii.gz
    ${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/002/t2w.nii.gz,${GANDLF_SYNTH_DATA_3DRAD_UNLABELED}/002/t1.nii.gz
    ```
3. [Construct the configuration file](https://github.com/mlcommons/GaNDLF-Synth/blob/main/docs/usage#customize-the-training) to help design the computation (training and inference) pipeline. You can use any model suitable for this task. An example file for this task can be found [here](https://github.com/mlcommons/GaNDLF-Synth/blob/main/testing/configs).
4. Now you are ready to [train your model](https://github.com/mlcommons/GaNDLF-Synth/blob/main/docs/usage#running-GaNDLF-Synth-traininginference).
5. Once the model is trained, you can use it to generate new images (or perform image-to-image reconstruction if you choose suitable model, such as VQVAE). [Add inference configuration](https://github.com/mlcommons/GaNDLF-Synth/blob/main/docs/usage#running-GaNDLF-Synth-traininginference) to the configuration file and run the inference.