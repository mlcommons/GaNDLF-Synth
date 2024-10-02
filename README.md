# GaNDLF-Synth

Presenting the **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for **Synth**esis (GaNDLF-Synth), a unified abstraction to train various synthesis algorithms in a zero/low code approach.

## Documentation

General documentation is in preparation, it will be **made available soon**.

Mixed precision training:
We currently support mixed precision training based on [lightning](https://pytorch-lightning.readthedocs.io/en/latest/advanced/mixed_precision.html). To use mixed precision, please set the "precision" field in the "compute" field. All available precision options can be found under the link above. 

```yaml
compute:
  precision: "16"        
```

Usage of distributed strategies:

- we currently support [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [DeepSpeed](https://www.deepspeed.ai/getting-started/). 
- To use ddp, just configure the number of nodes and type strategy name "ddp" under "compute" field in the config. Note that this is going to be used as default strategy if not specified and multiple GPUs are available.

```yaml
compute:
  num_devices: 2         # if not set, all GPUs available will 
  num_nodes: 2           # if not set, one node training is assumed
  strategy: "ddp"
  strategy_config: {}    # additional strategy specific kwargs, see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#lightning.pytorch.strategies.DDPStrategy

```

- For deepspeed, we leverage the original deepspeed library config to set the distributed parameters. To use deepspeed, configure the "compute" field as follows:

```yaml
compute:
  num_devices: 2         # if not set, all GPUs available will 
  num_nodes: 2           # if not set, one node training is assumed
  strategy: "deepspeed"
  strategy_config: 
    config: "path-to-deepspeed-config.json"    # path to the deepspeed config file
```
Details of this config file can be found in the deepspeed documentation here: https://www.deepspeed.ai/docs/config-json/
Please read further details in the Lightning guide: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html#custom-deepspeed-config.
Note that you will probably need to override the optimizer choice with one of optimized ones available in deepspeed. This optimizer (scheduler can be specified here too) will take precedence over the one specified in the base yaml config file.

## Citation

```
@misc{pati2024gandlfsynthframeworkdemocratizegenerative,
      title={GaNDLF-Synth: A Framework to Democratize Generative AI for (Bio)Medical Imaging}, 
      author={Sarthak Pati and Szymon Mazurek and Spyridon Bakas},
      year={2024},
      eprint={2410.00173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.00173}, 
}
```
