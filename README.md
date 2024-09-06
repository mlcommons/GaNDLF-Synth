Some basic info:
  - Deadline for package is 10th of October 2024 (MICCAI)
  - POC by the end of August 2024, allow new developer commits in September or so

Notes on distributed strategies:

- we currently support DDP and DeepSpeed
- for ddp, just configure the number of nodes and type strategy name "ddp" under "compute" field in the config
- for deepspeed, configure the number of nodes and type strategy name "deepspeed" under "compute" field in the config. Additionally, in the "strategy_config" dict under the "compute" field, specify the "config" field with the path to the deepspeed config json file. Details of this config file can be found in the deepspeed documentation here: https://www.deepspeed.ai/docs/config-json/ <br>
Also worth looking is the Lightning guide for that: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html#custom-deepspeed-config. Note that via config, you will probably need to override the optimizer choice with one of optimized ones availabe in deepspeed (look at the deepspeed documentation for that, link above). This optimizer (scheduler can be specified here too) will take precedence over the base yaml config file. 