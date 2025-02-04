
This file contains mid-level information regarding various parameters that can be leveraged to customize the training/inference in GaNDLF. To see the default parameters for certain fields, see the [default configs directory]("https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/parameter_defaults/").


## Training Parameters
The training parameters are defined in the configuration file. The following fields are supported:
```yaml
model_config:  # Configuration for the model (see below) - required
modality:  # Modality of the input data, either 'rad' or 'histo' - required
num_epochs:  # Number of epochs to train the model
batch_size:  # Batch size for training, validation and test dataloaders
data_preprocessing:  # Data preprocessing configuration (see below)
data_augmentation:  # Data augmentation configuration (see below)
data_postprocessing:  # Data postprocessing configuration (see below)
dataloader_config:  # Dataloaders configuration (see below)
inference_parameters:
  batch_size:  # Batch size for inference
  n_images_to_generate:  # Number of images to generate during inference, unused in image-to-image models that require input images. This field can be a single value or a dictionary containing the number of images to generate for each class, for example {"1": 10, "2": 20}.
save_model_every_n_epochs:  # Save checkpoint every n epochs
compute: {} # Distributed training and mixed precision configuration (see below)
```

## Model
Model configuration is expected to be in the following format:
```yaml
model_config:
    model_name:  # Name of the model to use
    labeling_paradigm:  # Labeling paradigm for the model, either 'unlabeled', 'patient', or 'custom'. Read further down for clarification on these three.
    architecture:  # Architecture of the model, customizing given model. Specifics are defined in the config of the given model.
    losses: # Loss functions to use (see below).
        - name:  # Name of the loss function
        - some_parameter: some_value
     # For models containing multiple losses (for example GANS), the losses are expected to be in the following format:
    losses:
        - discriminator: # Discriminator loss
            - name: # Name of the loss function
            - some_parameter: some_value
        - generator: # Generator loss
            - name: # Name of the loss function
            - some_parameter: some_value
    # Note that to use multiple losses, the model should be prepared via config to handle it via certain subloss name.
    - optimizers: # Optimizers to use (see below)
        - name:  # Name of the optimizer
        - some_parameter: some_value
    # For models containing multiple optimizers (for example GANS), the optimizers can be defined as losses above.
    - schedulers: # Schedulers to use (see below)
        - name:  # Name of the scheduler
        - some_parameter: some_value
    # For models containing multiple schedulers (for example GANS), the schedulers can be defined as losses above.
    - n_channels:  # Number of input channels
    - n_dimensions:  # Number of dimensions of the input data (2 or 3)
    - tensor_shape:  # Shape of the input tensor
    # This model config can support additional parameters that are specific to the model, for example:
    - save_eval_images_every_n_epochs:  # Save evaluation images every n epochs, useful to assess training progress of generative models. Implemented in i.e. DCGAN.
```
Regarding the "labeling_paradigm"
### Custom Labels

Custom labels are defined based on the folder structure specified by the user. For example, you can create directories such as `0`, `1`, `2`, etc., where each number represents a distinct class. These classes are arbitrary and can be assigned as per the user's requirements. Inside each class folder, you should organize **patient-specific subfolders** containing the corresponding data.

**Example Structure:**
```plaintext
/data
  ├── 0
  │    ├── patient_001
  │    └── patient_002
  ├── 1
  │    ├── patient_003
  │    └── patient_004
  └── 2
       ├── patient_005
       └── patient_006
```
### Patient-Level Labels

For patient-level labels, the folder structure consists of a main directory where each subfolder represents a specific patient. In this setup, each patient is treated as a separate class, similar to how labeling is handled in GANDLF.

**Example Structure:**
```plaintext
/data
  ├── patient_001
  ├── patient_002
  ├── patient_003
  └── patient_004
```

### Unlabeled Data

For unlabeled data, you are required to maintain a patient-wise folder structure, similar to the patient-level labeling setup. However, in this case, class labels are ignored. The data is simply loaded without any label association.

**Example Structure:**
```plaintext
/data
  ├── patient_001
  ├── patient_002
  ├── patient_003
  └── patient_004
```


## Optimizers
GaNDLF-Synth interfaces GaNDLF core framework for optimizers. See the [optimizers directory](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/optimizers/__init__.py) for available optimizers. They support optimizer-specific configurable parameters, interfacing [Pytorch Optimizers](https://pytorch.org/docs/stable/optim.html).

## Schedulers
GaNDLF-Synth interfaces GaNDLF core framework for schedulers. See the [schedulers directory](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/schedulers/__init__.py) for available schedulers. They support scheduler-specific configurable parameters, interfacing [Pytorch Schedulers](https://pytorch.org/docs/stable/optim.html).

## Losses
GaNDLF-Synth supports multiple loss functions. See the [losses directory](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/losses/__init__.py) for available loss functions. They support loss-specific configurable parameters, interfacing [Pytorch Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions).

## Dataloader configs
GaNDLF-Synth supports separate dataloader parameters for training, validation, test and inference dataloaders. They support configurable parameters, interfacing [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html). The following fields are supported:
```yaml
dataloader_config:  # Dataloaders configuration (see below)
  shuffle:  # Whether to shuffle the data
  num_workers:  # Number of processes spawned to load the data
  pin_memory:  # Whether to pin the memory for CPU-GPU transfer
  timeout:  # Timeout for the dataloader processes
  prefetch_factor:  # Number of batches to prefetch by each worker
  persistent_workers:  # Whether to keep the worker processes alive between epochs
```
The fields for specific dataloaders are expected to be in the following format:
```yaml
dataloader_config:
    train:
        - some_parameter: some_value
    val:
        - some_parameter: some_value
    test:
        - some_parameter: some_value
    inference:
        - some_parameter: some_value
```
If given dataloader is not configured explicitly, the default values are used (see above).

## Data preprocessing
GaNDLF-Synth interfaces GaNDLF core framework for data preprocessing. To see available data preprocessing options, see [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/preprocessing/__init__.py).
Separate preprocessing parameters can be defined for each dataloader (train, val, test, inference) as follows:
```yaml
data_preprocessing:
    train:
        - some_transform: some_value
    val:
        - some_transform: some_value
    test:
        - some_transform: some_value
    inference:
        - some_transform: some_value
```

## Data Augmentation
GaNDLF-Synth interfaces GaNDLF core framework for data augmentation. To see available data augmentation options, see [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/augmentation/__init__.py) Augmentations are applied only to the training dataloader. 
```yaml
data_augmentation:
    train:
        - some_transform: some_value
```


## Post processing
GaNDLF-Synth interfaces GaNDLF core framework for post processing. To see available post processing options, see [here](https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/data/post_process/__init__.py). Post-processing is applied only to the inference dataloader.
```yaml
data_postprocessing:
    inference:
        - some_transform: some_value
```

## Distributed Training
For detalis on using distributed training, see the [usage page](./usage.md#parallelize-the-training-and-inference).


