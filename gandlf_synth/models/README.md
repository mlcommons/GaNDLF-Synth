When adding new model, please do the following:
1. Define new model config in a file named {model_name_config.py} in this directory.
2. Add this config to the AVAILABLE_MODELS_CONFIGS dictionary in
the ModelConfigFactory class in gandlf_synth/models/configs/model_config_factory.py.
3. Define the new model architecture in a file named {model_name.py} in this directory bu subclassing base_model.py.
4. In ./modules directory, define the new model's module (all related logic) in a file named {model_name_module.py} by subclassing
SynthesisModule from the ./modules/module_abc.py.