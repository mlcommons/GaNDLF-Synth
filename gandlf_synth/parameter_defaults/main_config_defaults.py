from gandlf_synth.parameter_defaults.dataloader_defaults import (
    DATALOADER_CONFIG_DEFAULTS,
)

REQUIRED_PARAMETERS = ["model_config", "modality"]


BASIC_PARAMETER_DEFAULTS = {
    "num_epochs": 100,  # total number of epochs to train
    "batch_size": 1,  # default batch size of training
    "data_postprocessing": {},  # default data postprocessing
    "data_preprocessing": {},  # default data preprocessing
    "data_augmentation": {},  # default data augmentation
    "dataloader_config": DATALOADER_CONFIG_DEFAULTS,  # dataloader configuration
    "save_model_every_n_epochs": -1,  # save model every n epochs
    "compute": {},  # compute parameters, please refer to the README file for more information
}
