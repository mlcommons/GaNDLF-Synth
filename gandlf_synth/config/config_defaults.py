REQUIRED_PARAMETERS = ["model_config", "modality"]

TRAIN_LOADER_CONFIG = {
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
    "shuffle": True,
}

VALIDATION_LOADER_CONFIG = {
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
    "shuffle": False,
}

TEST_LOADER_CONFIG = {
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
    "shuffle": False,
}

INFER_LOADER_CONFIG = {
    "num_workers": 0,
    "pin_memory": False,
    "drop_last": False,
    "shuffle": False,
}

DATALOADER_CONFIG = {
    "train": TRAIN_LOADER_CONFIG,
    "validation": VALIDATION_LOADER_CONFIG,
    "test": TEST_LOADER_CONFIG,
    "infer": INFER_LOADER_CONFIG,
}


PARAMETER_DEFAULTS = {
    "amp": False,  # automatic mixed precision
    "verbose": False,  # general application verbosity
    "save_training": False,  # save outputs during training
    "save_output": False,  # save outputs during validation/testing
    "in_memory": False,  # pin data to cpu memory
    "num_epochs": 100,  # total number of epochs to train
    "patience": 0,  # number of epochs to wait for performance improvement
    "batch_size": 1,  # default batch size of training
    "clip_grad": None,  # clip_gradient value
    "track_memory_usage": False,  # default memory tracking
    "memory_save_mode": False,  # default memory saving, if enabled, resize/resample will save files to disk
    "print_rgb_label_warning": True,  # print rgb label warning
    "data_postprocessing": {},  # default data postprocessing
    "data_preprocessing": {},  # default data preprocessing
    "data_augmentation": {},  # default data augmentation
    "previous_parameters": None,  # previous parameters to be used for resuming training and perform sanity checking
    "dataloader_config": DATALOADER_CONFIG,  # dataloader configuration
}
