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

DATALOADER_CONFIG_DEFAULTS = {
    "train": TRAIN_LOADER_CONFIG,
    "validation": VALIDATION_LOADER_CONFIG,
    "test": TEST_LOADER_CONFIG,
    "infer": INFER_LOADER_CONFIG,
}
