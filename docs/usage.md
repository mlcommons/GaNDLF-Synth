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

### Anonymize Data

A major reason why one would want to anonymize data is to ensure that trained models do not inadvertently encode protected health information [[1](https://doi.org/10.1145/3436755),[2](https://doi.org/10.1038/s42256-020-0186-1)]. For this task, one may use base GaNDLF. GaNDLF can anonymize one or multiple images using the `gandlf anonymizer` command as follows:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf anonymizer
  # -h, --help         Show help message and exit
  -c ./samples/config_anonymizer.yaml \ # anonymizer configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./input_dir_or_file \ # input directory containing series of images to anonymize or a single image
  -o ./output_dir_or_file # output directory to save anonymized images or a single output image file (for a DICOM to NIfTi conversion specify a .nii.gz file)
```
### Cleanup/Harmonize/Curate Data

It is **highly** recommended that the dataset you want to train/infer on has been harmonized. The following requirements should be considered:

- Registration
    - Within-modality co-registration [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)].
    - **OPTIONAL**: Registration of all datasets to patient atlas, if applicable [[1](https://doi.org/10.1109/TMI.2014.2377694), [2](https://doi.org/10.1038/sdata.2017.117), [3](https://arxiv.org/abs/1811.02629)].
- **Intensity harmonization**: Same intensity profile, i.e., normalization [[4](https://doi.org/10.1016/j.nicl.2014.08.008), [5](https://visualstudiomagazine.com/articles/2020/08/04/ml-data-prep-normalization.aspx), [6](https://developers.google.com/machine-learning/data-prep/transform/normalization), [7](https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0)]. GaNDLF offers [multiple options](#customize-the-training) for intensity normalization, including Z-scoring, Min-Max scaling, and Histogram matching. 
- **Resolution harmonization**: Ensures that the images have *similar* physical definitions (i.e., voxel/pixel size/resolution/spacing). An illustration of the impact of voxel size/resolution/spacing can be found [here](https://upenn.box.com/v/spacingsIssue), and it is encourage to read [this article](https://www.nature.com/articles/s41592-020-01008-z#:~:text=of%20all%20images.-,Resampling,-In%20some%20datasets) to added context on how this issue impacts a deep learning pipeline. This functionality is available via [GaNDLF's preprocessing module](#customize-the-training).

Recommended tools for tackling all aforementioned curation and annotation tasks: 
- [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk) 
- [Federated Tumor Segmentation (FeTS) Front End](https://github.com/FETS-AI/Front-End)
- [3D Slicer](https://www.slicer.org)
- [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)

### Offline Patch Extraction (for histology images only)

GaNDLF can be used to convert a Whole Slide Image (WSI) with or without a corresponding label map to patches/tiles using GaNDLF’s integrated patch miner, which would need the following files:

1. A configuration file that dictates how the patches/tiles will be extracted. A sample configuration to extract patches is presented [here](https://github.com/mlcommons/GaNDLF/blob/master/samples/config_getting_started_segmentation_histo2d_patchExtraction.yaml). The options that the can be defined in the configuration are as follows:
     - `patch_size`: defines the size of the patches to extract, should be a tuple type of integers (e.g., `[256,256]`) or a string containing patch size in microns (e.g., `[100m,100m]`). This parameter always needs to be specified.
     - `scale`: scale at which operations such as tissue mask calculation happens; defaults to `16`.
     - `num_patches`: defines the number of patches to extract, use `-1` to mine until exhaustion; defaults to `-1`.
     - `value_map`: mapping RGB values in label image to integer values for training; defaults to `None`.
     - `read_type`: either `random` or `sequential` (latter is more efficient); defaults to `random`.
     - `overlap_factor`: Portion of patches that are allowed to overlap (`0->1`); defaults to `0.0`.
     - `num_workers`: number of workers to use for patch extraction (note that this does not scale according to the number of threads available on your machine); defaults to `1`.
2. A CSV file with the following columns:
     - `SubjectID`: the ID of the subject for the WSI
     - `Channel_0`: the full path to the WSI file which will be used to extract patches
     - `Label`: (optional) full path to the label map file

Once these files are present, the patch miner can be run using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf patch-miner \ 
  # -h, --help         Show help message and exit
  -c ./exp_patchMiner/config.yaml \ # patch extraction configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./exp_patchMiner/input.csv \ # data in CSV format 
  -o ./exp_patchMiner/output_dir/ # output directory
```

### Running preprocessing before training/inference (optional)

Running preprocessing before training/inference is optional, but recommended. It will significantly reduce the computational footprint during training/inference at the expense of larger storage requirements. Use the following command, which will save the processed data in `./experiment_0/output_dir/` with a new data CSV and the corresponding model configuration:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf preprocess \
  # -h, --help         Show help message and exit
  -c ./experiment_0/model.yaml \ # model configuration - needs to be a valid YAML (check syntax using https://yamlchecker.com/)
  -i ./experiment_0/train.csv \ # data in CSV format 
  -o ./experiment_0/output_dir/ # output directory
```


## Constructing the Data CSV

This application can leverage multiple channels/modalities for training while using a multi-class segmentation file. The expected format is shown as an example in [samples/sample_train.csv](https://github.com/mlcommons/GaNDLF/blob/master/samples/sample_train.csv) and needs to be structured with the following header format (which shows a CSV with `N` subjects, each having `X` channels/modalities that need to be processed):

```csv
SubjectID,Channel_0,Channel_1,...,Channel_X,Label
001,/full/path/001/0.nii.gz,/full/path/001/1.nii.gz,...,/full/path/001/X.nii.gz,/full/path/001/segmentation.nii.gz
002,/full/path/002/0.nii.gz,/full/path/002/1.nii.gz,...,/full/path/002/X.nii.gz,/full/path/002/segmentation.nii.gz
...
N,/full/path/N/0.nii.gz,/full/path/N/1.nii.gz,...,/full/path/N/X.nii.gz,/full/path/N/segmentation.nii.gz
```

**Notes:**

- `Channel` can be substituted with `Modality` or `Image`
- `Label` can be substituted with `Mask` or `Segmentation` and is used to specify the annotation file for segmentation models
- For classification/regression, add a column called `ValueToPredict`. Currently, we are supporting only a single value prediction per model.
- Only a single `Label` or `ValueToPredict` header should be passed 
    - Multiple segmentation classes should be in a single file with unique label numbers.
    - Multi-label classification/regression is currently not supported.

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

### Using the `gandlf split-csv` command

To split the data CSV into training, validation, and testing CSVs, the `gandlf split-csv` script can be used. The following command shows how the script works:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf split-csv \
  # -h, --help         Show help message and exit
  -i ./experiment_0/train_data.csv \ # output CSV from the `gandlf construct-csv` script
  -c $gandlf_config \ # the GaNDLF config (in YAML) with the `nested_training` key specified to the folds needed
  -o $output_dir # the output directory to save the split data
```


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

### Special notes for Inference for Histology images

- If you trying to perform inference on pre-extracted patches, please change the `modality` key in the configuration to `rad`. This will ensure the histology-specific pipelines are not triggered.
- However, if you are trying to perform inference on full WSIs, `modality` should be kept as `histo`.


## Generate Metrics 

GaNDLF provides a script to generate metrics after an inference process is done.The script can be used as follows:

```bash
# continue from previous shell
(venv_gandlf) $> gandlf generate-metrics \
  # -h, --help         Show help message and exit
  # -v, --version      Show program's version number and exit.
  -c , --config       The configuration file (contains all the information related to the training/inference session)
  -i , --input-data    CSV file that is used to generate the metrics; should contain 3 columns: 'SubjectID,Target,Prediction'
  -o , --output-file   Location to save the output dictionary. If not provided, will print to stdout.
```

Once you have your CSV in the specific format, you can pass it on to generate the metrics. Here is an example for segmentation:

```csv
SubjectID,Target,Prediction
001,/path/to/001/target.nii.gz,/path/to/001/prediction.nii.gz
002,/path/to/002/target.nii.gz,/path/to/002/prediction.nii.gz
...
```

Similarly, for classification or regression (`A`, `B`, `C`, `D` are integers for classification and floats for regression):

```csv
SubjectID,Target,Prediction
001,A,B
002,C,D
...
```

To generate image to image metrics for synthesis tasks (including for the BraTS synthesis tasks [[1](https://www.synapse.org/#!Synapse:syn51156910/wiki/622356), [2](https://www.synapse.org/#!Synapse:syn51156910/wiki/622357)]), ensure that the config has `problem_type: synthesis`, and the CSV can be in the same format as segmentation (note that the `Mask` column is optional):

```csv
SubjectID,Target,Prediction,Mask
001,/path/to/001/target_image.nii.gz,/path/to/001/prediction_image.nii.gz,/path/to/001/brain_mask.nii.gz
002,/path/to/002/target_image.nii.gz,/path/to/002/prediction_image.nii.gz,/path/to/002/brain_mask.nii.gz
...
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
