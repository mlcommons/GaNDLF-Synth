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

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown in example CSVs in [samples directory](https://github.com/mlcommons/GaNDLF-Synth/blob/main/samples) for both labeled and unlabeled data. The CSV file needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

#### Unlabeled Data
```csv
Channel_0,Channel_1,...,Channel_X
$ROOT-PATH-TO-DATA-FOLDER/1/1.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/1/2.nii.gz,...,
$ROOT-PATH-TO-DATA-FOLDER/2/1.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/2/2.nii.gz,...,
$ROOT-PATH-TO-DATA-FOLDER/3/1.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/3/2.nii.gz,...,
...
```

#### Labeled Data
```csv
Channel_0,Channel_1,Label,LabelMapping
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/1/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/1/t1.nii.gz,0,$CLASS-FOLDER-NAME-1
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/2/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/2/t1.nii.gz,0,$CLASS-FOLDER-NAME-1
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/3/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-1/3/t1.nii.gz,0,$CLASS-FOLDER-NAME-1
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/1/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/1/t1.nii.gz,1,$CLASS-FOLDER-NAME-2
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/2/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/2/t1.nii.gz,1,$CLASS-FOLDER-NAME-2
$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/3/t2w.nii.gz,$ROOT-PATH-TO-DATA-FOLDER/$CLASS-FOLDER-NAME-2/3/t1.nii.gz,1,$CLASS-FOLDER-NAME-2

...
```

**Notes:**

- For labeled data, the CSV has additonal columns for the labels assigned to given set of channels. It also has a column for the label mapping, showing the class name assigned to the label value.

### Using the `gandlf-synth construct-csv` command

To make the process of creating the CSV easier, we have provided a `gandlf-synth construct-csv` command. The data has to be arranged in different formats, depeinding on labeling paradigm. Modality names are used as examples.

#### Unlabeled Data

```bash
$DATA_DIRECTORY
│
└───Patient_001 
│   │ Patient_001_brain_t1.nii.gz
│   │ Patient_001_brain_t1ce.nii.gz
│   │ Patient_001_brain_t2.nii.gz
│   │ Patient_001_brain_flair.nii.gz
│   
│
└───Patient_002 
│   │ Patient_002_brain_t1.nii.gz
│   │ Patient_002_brain_t1ce.nii.gz
│   │ Patient_002_brain_t2.nii.gz
│   │ Patient_002_brain_flair.nii.gz
│   
...
|
└───JaneDoe # Patient name can be different
│   │ randomFileName_0_t1.nii.gz 
│   │ randomFileName_1_t1ce.nii.gz
│   │ randomFileName_2_t2.nii.gz
│   │ randomFileName_3_flair.nii.gz
│   │ randomFileName_seg.nii.gz 
│
...
```

#### Custom Class Labeled Data

```bash
$DATA_DIRECTORY
│
└───Class_1 # the class name can be different
│   │
│   └───Patient_001
│   │   │ Patient_001_brain_t1.nii.gz
│   │   │ Patient_001_brain_t1ce.nii.gz
│   │   │ Patient_001_brain_t2.nii.gz
│   │   │ Patient_001_brain_flair.nii.gz
│   │
│   └───Patient_002
│       │ Patient_002_brain_t1.nii.gz
│       │ Patient_002_brain_t1ce.nii.gz
│       │ Patient_002_brain_t2.nii.gz
│       │ Patient_002_brain_flair.nii.gz
│
└───Class_2
│   │
│   └───Patient_003
│   │   │ Patient_003_brain_t1.nii.gz
│   │   │ Patient_003_brain_t1ce.nii.gz
│   │   │ Patient_003_brain_t2.nii.gz
│   │   │ Patient_003_brain_flair.nii.gz
│   │
...
|
└───Class_N
    │
    └───Patient_M
        │ Patient_M_brain_t1.nii.gz
        │ Patient_M_brain_t1ce.nii.gz
        │ Patient_M_brain_t2.nii.gz
        │ Patient_M_brain_flair.nii.gz
```


#### Per Patient Labeled Data

The structure is similar to the unlabeled data, the labels are assigned per patient. 


The following command shows how the script works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf-synth construct-csv \
  # -h, --help         Show help message and exit
  -i $DATA_DIRECTORY # this is the main data directory 
  -ch _t1.nii.gz,_t1ce.nii.gz,_t2.nii.gz,_flair.nii.gz \ # an example image identifier for structural brain MR sequences for BraTS, and can be changed based on your data. In the simplest case of a single modality, a ".nii.gz" will suffice
  -l 'unlabeled' \ # labeling paradigm, can be 'unlabeled', 'patient', or 'custom'
  -o ./experiment_0/train_data.csv # output CSV to be used for training
```

## Customize the Training

Adapting GaNDLF to your needs boils down to modifying a YAML-based configuration file which controls the parameters of training and inference. Below is a list of available samples for users to start as their baseline for further customization:

- [DDPM (unlabeled paradigm)](https://github.com/mlcommons/GaNDLF-Synth/blob/main/samples/example_config_ddpm_unlabeled.yaml)
- [VQVAE (unlabeled paradigm)](https://github.com/mlcommons/GaNDLF-Synth/blob/main/samples/example_config_vqvae_unlabeled.yaml)

<!-- To find **all the parameters** a GaNDLF config may modify, consult the following file: 
- [All available options](https://github.com/mlcommons/GaNDLF-Synth/blob/main/samples/config_all_options.yaml) -->

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
  -dt ./experiment_0/train.csv \ # main data CSV used for training (or inference if performing image-to-image reconstruction)
  -m-dir ./experiment_0/model_dir/ \ # model directory where the output of the training will be stored, created if not present
  --t \ # enable training (if not enabled, inference is performed)
  # -v-csv ./experiment_0/val.csv \ # [optional] validation data CSV (if the model performs validation step)
  # -t-csv ./experiment_0/test.csv \ # [optional] testing data CSV (if the model performs testing step)
  # -vr 0.1 \ # [optional] ratio of validation data to extract from the training data CSV. If -v-csv flag is set, this is ignored
  # -tr 0.1 \ # [optional] ratio of testing data to extract from the training data CSV. If -t-csv flag is set, this is ignored
  # -i-dir ./experiment_0/inference_dir/ \ # [optional] inference directory where the output of the inference will be stored, created if not present. Used only if inference is enabled
  # -ckpt-path ./experiment_0/model_dir/checkpoint.ckpt \ # [optional] path to the checkpoint file to resume training from or to use for inference. If not provided, the latest (or best) checkpoint is used when resuming training or performing inference
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

Once your model is trained, you should see the following outputin the model directory:

```bash
# continue from previous shell
(venv_gandlf) $> ls ./experiment_0/model_dir/
checkpoints/ # directory containing all the checkpoints
training_logs/ # directory containing all the training logs
eval_images/ # optionally created - if model was configured to periodically save evaluation images after N epochs (only for 2D runs)
parameters.pkl # the used configuration file
training_manager.log # global log file of the training manager
```
If you performed inference, the inference directory will contain the following:

```bash
# continue from previous shell
(venv_gandlf) $> ls ./experiment_0/inference_dir/
inference_output/ # directory containing all the inference output (synthesized images)
ingerence_logs/ # directory containing all the inference logs from the trainer
inference_manager.log # global log file of the inference manager
```