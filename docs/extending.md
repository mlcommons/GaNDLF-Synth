## Environment

Before starting to work on the code-level on GaNDLF, please follow the [instructions to install GaNDLF-Synth from sources](./setup.md). Once that's done, please verify the installation using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> 
# you should be in the "GaNDLF" git repo
(venv_gandlf) $> gandlf-synth verify-install
```


## Submodule flowcharts

- The following flowcharts are intended to provide a high-level overview of the different submodules in GaNDLF-Synth. 
- Navigate to the `README.md` file in each submodule folder for details.
- Some flowcharts are still in development and may not be complete/present.

## Overall Architecture

- Command-line parsing: [gandlf-synth run](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/entrypoints/run.py)
- [Config Manager](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/config_manager.py): 
    - Handles configuration parsing
    - Provides configuration to other modules
- [Training Manager](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/training_manager.py): 
    - Main entry point from CLI
    - Handles training functionality
- [Inference Manager](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/inference_manager.py): 
    - Handles inference functionality 
    - Main entry point from CLI
    - Performs actual inference 


## Dependency Management

To update/change/add a dependency in [setup](https://github.com/mlcommons/GaNDLF-Synth/blob/main/setup.py), please ensure **at least** the following conditions are met:

- The package is being [actively maintained](https://opensource.com/life/14/1/evaluate-sustainability-open-source-project).
- The new dependency is being testing against the **minimum python version** supported by GaNDLF-Synth (see the `python_requires` variable in [setup](https://github.com/mlcommons/GaNDLF-Synth/blob/main/setup.py)).
- It does not clash with any existing dependencies.

## Adding Models

- Create the model in the new file in [architectures](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/models/architectures) folder.
- Make sure the new class inherits from [ModelBase](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/models/architectures/base_model.py) class.
- Create new `LightningModule` that will implement the training, validation, testing and inference logic. Make sure it inherits from [SynthesisModule] (https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/models/module_abc.py) class and implements necessary abstract methods.
- Add the new model to the [ModuleFactory](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/models/module_factory.py) `AVAILABE_MODULES` dictionary. Note that the key in this dictionary should follow the naming convention: `labeling-paradigm_model-name`, and needs to match the key in the model config factory that we will create in the next steps.
- Create the model config file in the [configs](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/configs) folder.
- Implement the model configuration class that will parse model's configuration when creating model instance. Make sure it inherits from [AbstractModelConfig](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/configs/config_abc.py) class.
- Add the new model configuration class to the [ModelConfigFactory](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/configs/model_config_factory.py) `AVAILABLE_MODEL_CONFIGS` dictionary. Note that the model config key in this dictionary should follow the naming convention: `labeling-paradigm_model-name`, and needs to match the key in the model factory that we created in the previous steps.

## Adding Dataloading Functionalities

GaNDLF-Synth handles dataloading for specific labeling paradigms in separate abstractions. If you wish to modify the dataloading functionality, please refer to the following modules:

- [Datasets](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/data/datasets.py): Contains the dataset classes for different labeling paradigms.
- [Dataset Factories](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/data/datasets_factory.py): Contains the factory class that creates the dataset instance based on the configuration.
- [Dataloaders Factories](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/data/dataloaders_factory.py): Contains the factory class that creates the dataloader instance based on the configuration.

Remember to add new datasets and dataloaders to the respective factory classes. For some cases, modifications of the training or inference logic may be required to accommodate the new dataloading functionality (see below).

## Adding Training Functionality


- For changes at the level of single training, validation, or test steps, modify the specific functions of a given module.
- For changes at the level of the entire training loop, modify the [Training Manager](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/training_manager.py). The main training loop is handled via `Trainer` class of `Pytorch Lightning` - please refer to the [Pytorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) for more details.


## Adding Inference Functionality

- For changes at the level of single inference steps, modify the specific functions of a given module. Note that for inference, special dataloaders are used to load the data in the required format.
- For changes at the level of the entire inference loop, modify the [Inference Manager](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/inference_manager.py). The main inference loop is handled via `Trainer` class of `Pytorch Lightning` - please refer to the [Pytorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) for more details.

## Adding new CLI command

Example: `gandlf-synth run` [CLI command](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/entrypoints/construct_csv.py)
- Implement function and wrap it with `@click.command()` + `@click.option()`
- Add it to `cli_subommands` [dict](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/entrypoints/subcommands.py)
The command would be available under `gandlf-synth your-subcommand-name` CLI command.


## Update parameters
For any new feature that is configurable via config, please ensure the corresponding option in the ["extending" section of this documentation](./extending.md) is added, so that others can review/use/extend it as needed.

## Update Tests

Once you have made changes to functionality, it is imperative that the unit tests be updated to cover the new code. Please see the [full testing suite](https://github.com/mlcommons/GaNDLF-Synth/blob/main/gandlf_synth/testing/tests/) for details and examples. Note that tests are split into different categories, each having its own file in the aforementioned folder:
- `test_modules.py`: module-specific tests
- `test_generic.py`: global features tests
- `entrypoints/`: tests for specific CLI commands

## Run Tests

### Prerequisites

Tests are using [sample data](https://drive.google.com/uc?id=12utErBXZiO_0hspmzUlAQKlN9u-manH_), which gets downloaded and prepared automatically when you run unit tests. Prepared data is stored at `GaNDLF-Synth/testing/data/` automatically the first time test are ran. However, you may want to download & explore data by yourself.

### Unit tests

Once you have the virtual environment set up, tests can be run using the following command:

```bash
# continue from previous shell
(venv_gandlf) $> pytest --device cuda # can be cuda or cpu, defaults to cpu
```

Any failures will be reported in the file `GaNDLF-Synth/testing/failures.log`.


### Code coverage

The code coverage for the unit tests can be obtained by the following command:

```bash
# continue from previous shell
(venv_gandlf) $> coverage run -m pytest --device cuda; coverage report -m
```
