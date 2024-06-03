from gandlf_synth.parameter_defaults.dataloader_defaults import (
    DATALOADER_CONFIG_DEFAULTS,
)

REQUIRED_PARAMETERS = ["model_config", "modality"]


BASIC_PARAMETER_DEFAULTS = {
    "verbose": False,  # general application verbosity
    "save_training": False,  # save outputs during training
    "save_output": False,  # save outputs during validation/testing
    "in_memory": False,  # pin data to cpu memory
    "num_epochs": 100,  # total number of epochs to train
    "batch_size": 1,  # default batch size of training
    "track_memory_usage": False,  # default memory tracking
    "memory_save_mode": False,  # default memory saving, if enabled, resize/resample will save files to disk
    "print_rgb_label_warning": True,  # print rgb label warning
    "data_postprocessing": {},  # default data postprocessing
    "data_preprocessing": {},  # default data preprocessing
    "data_augmentation": {},  # default data augmentation
    "previous_parameters": None,  # previous parameters to be used for resuming training and perform sanity checking
    "dataloader_config": DATALOADER_CONFIG_DEFAULTS,  # dataloader configuration
}
