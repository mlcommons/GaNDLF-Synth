## Introduction

The usual DL workflow consists of the following steps:

1. Prepare the data
2. Split data into training, validation, and testing
3. Customize the training parameters
4. Train the model
5. Perform inference

GaNDLF-Synth supports all of these steps. Some of the steps are treated as optional due to the nature of the generation tasks. For example, sometimes you do not want to split the data into training, validation, and testing, but rather use all the data for training. GaNDLF-Synth provides the necessary tools to perform these tasks, using both custom features and the ones provided by GaNDLF. We describe all the functionalities in the following sections. For more details on the functionalities directly from GaNDLF, please refer to the [GaNDLF documentation](https://docs.mlcommons.org/GaNDLF).

## Installation

Please follow the [installation instructions](./setup.md#installation) to install GaNDLF-Synth.

## Preparing the Data
## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/mlcommons/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

```csv
Channel_0,Channel_1,...,Channel_X
/full/path/001/0.nii.gz,/full/path/001/1.nii.gz,...
/full/path/002/0.nii.gz,/full/path/002/1.nii.gz,...
...
/full/path/N/0.nii.gz,/full/path/N/1.nii.gz,...,
```

**Notes:**

- For labeled data, the CSV will have additonal columns for the labels assigned to given set of channels.

### Using the `gandlf construct-csv` command

To make the process of creating the CSV easier, we have provided a `gandlf construct-csv` command. This script works when the data is arranged in the following format (example shown of the data directory arrangement from the [Brain Tumor Segmentation (BraTS) Challenge](https://www.synapse.org/brats)):

```bash
$DATA_DIRECTORY
│
└───Patient_001 # this is constructed from the ${PatientID} header of CSV
│   │ Patient_001_brain_t1.nii.gz
│   │ Patient_001_brain_t1ce.nii.gz
│   │ Patient_001_brain_t2.nii.gz
│   │ Patient_001_brain_flair.nii.gz
│   │ Patient_001_seg.nii.gz # optional for segmentation tasks
│
└───Patient_002 # this is constructed from the ${PatientID} header of CSV
│   │ Patient_002_brain_t1.nii.gz
│   │ Patient_002_brain_t1ce.nii.gz
│   │ Patient_002_brain_t2.nii.gz
│   │ Patient_002_brain_flair.nii.gz
│   │ Patient_002_seg.nii.gz # optional for segmentation tasks
│
└───JaneDoe # this is constructed from the ${PatientID} header of CSV
│   │ randomFileName_0_t1.nii.gz # the string identifier needs to be the same for each modality
│   │ randomFileName_1_t1ce.nii.gz
│   │ randomFileName_2_t2.nii.gz
│   │ randomFileName_3_flair.nii.gz
│   │ randomFileName_seg.nii.gz # optional for segmentation tasks
│
...
```

The following command shows how the script works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf construct-csv \
  # -h, --help         Show help message and exit
  -i $DATA_DIRECTORY # this is the main data directory 
  -c _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # an example image identifier for 4 structural brain MR sequences for BraTS, and can be changed based on your data. In the simplest case of a single modality, a ".nii.gz" will suffice
  -l _seg.nii.gz \ # an example label identifier - not needed for regression/classification, and can be changed based on your data
  -o ./experiment_0/train_data.csv # output CSV to be used for training
```

**Notes**:

- For classification/regression, add a column called `ValueToPredict`. Currently, we support only a single value prediction per model.
- `SubjectID` or `PatientName` is used to ensure that the randomized split is done per-subject rather than per-image.
- For data arrangement different to what is described above, a customized script will need to be written to generate the CSV, or you can enter the data manually into the CSV. 


## Customize the Training

Adapting GaNDLF to your needs boils down to modifying a YAML-based configuration file which controls the parameters of training and inference. Below is a list of available samples for users to start as their baseline for further customization:

- [Segmentation example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_segmentation_brats.yaml)
- [Regression example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_regression.yaml)
- [Classification example](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_classification.yaml)

To find **all the parameters** a GaNDLF config may modify, consult the following file: 
- [All available options](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_all_options.yaml)

**Notes**: 

- More details on the configuration options are available in the [customization page](customize.md).
- Ensure that the configuration has valid syntax by checking the file using any YAML validator such as [yamlchecker.com](https://yamlchecker.com/) or [yamlvalidator.com](https://yamlvalidator.com/) **before** trying to train.



## Running GaNDLF (Training/Inference)

You can use the following code snippet to run GaNDLF:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf run \
  # -h, --help         Show help message and exit
  # -v, --version      Show program's version number and exit.
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -m ./experiment_0/model_dir/ \ # model directory (i.e., the `model-dir`) where the output of the training will be stored, created if not present
  --train \ # --train/-t or --infer
  -d cuda # ensure CUDA_VISIBLE_DEVICES env variable is set for GPU device, use 'cpu' for CPU workloads
  # -rt , --reset # [optional] completely resets the previous run by deleting `model-dir`
  # -rm , --resume # [optional] resume previous training by only keeping model dict in `model-dir`
```

## Parallelize the Training

### Multi-GPU training

GaNDLF enables relatively straightforward multi-GPU training. Simply set the `CUDA_VISIBLE_DEVICES` environment variable to the list of GPUs you want to use, and pass `cuda` as the device to the `gandlf run` command. For example, if you want to use GPUs 0, 1, and 2, you would set `CUDA_VISIBLE_DEVICES=0,1,2` [[ref](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/)] and pass `-d cuda` to the `gandlf run` command.

### Distributed training

Distributed training is a more difficult problem to address, since there are multiple ways to configure a high-performance computing cluster (SLURM, OpenHPC, Kubernetes, and so on). Owing to this discrepancy, we have ensured that GaNDLF allows multiple training jobs to be submitted in relatively straightforward manner using the command line inference of each site’s configuration. Simply populate the `parallel_compute_command` in the [configuration](#customize-the-training) with the specific command to run before the training job, and GaNDLF will use this string to submit the training job. 


## Expected Output(s)

### Training

Once your model is trained, you should see the following output:

```bash
# continue from previous shell
(venv_gandlf) $> ls ./experiment_0/model_dir/
data_${cohort_type}.csv  # data CSV used for the different cohorts, which can be either training/validation/testing
data_${cohort_type}.pkl  # same as above, but in pickle format
logs_${cohort_type}.csv  # logs for the different cohorts that contain the various metrics, which can be either training/validation/testing
${architecture_name}_best.pth.tar # the best model in native PyTorch format
${architecture_name}_latest.pth.tar # the latest model in native PyTorch format
${architecture_name}_initial.pth.tar # the initial model in native PyTorch format
${architecture_name}_initial.{onnx/xml/bin} # [optional] if ${architecture_name} is supported, the graph-optimized best model in ONNX format
# other files dependent on if training/validation/testing output was enabled in configuration
```

### Inference

# TODO
