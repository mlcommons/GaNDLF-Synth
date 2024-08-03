When adding new model, please do the following:
1. Define new model config in a file named {model_name_config.py} in gandlf_synth/models/configs directory by subclassing AbstractModelConfig in the config_abc.py file in the same directory. If setting default values for the config, define them in the `MODEL_SPECIFIC_DEFAULT_PARAMS` (for params related to some general configuration of using your model) and `ARCHITECTURE_DEFAULT_PARAMS` (for params related to the architecture of your model).
2. Add this config to the AVAILABLE_MODELS_CONFIGS dictionary in
the ModelConfigFactory class in gandlf_synth/models/configs/model_config_factory.py.
3. Define the new model architecture in a file named {modelname.py} in gandlf_synth/models/architectures directory by subclassing BaseModel from the base_model.py file in the same directory.
4. In gandlf_synth/models/modules directory, define the new model's module (all related logic) in a file named {modelname_module.py} by subclassing
SynthesisModule from the gandlf_synth/modules/module_abc.py.
5. Add the new model's module to the AVAILABLE_MODULES dictionary in the ModuleFactory class in gandlf_synth/modules/module_factory.py