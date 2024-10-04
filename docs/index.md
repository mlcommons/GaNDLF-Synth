# GaNDLF

The **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework - **synth**esis (GaNDLF-synth) for reproducible and automated deep generative modeling in medical imaging.

## Why use GaNDLF-synth?

GaNDLF-synth was developed to lower the barrier to AI, enabling reproducibility, translation, and deployment regarding usage of generative models in medical imaging.
It is an extension of the [GaNDLF](https://docs.mlcommons.org/GaNDLF/) framework, which is a part of the [MLCommons](https://mlcommons.org/) initiative.
GaNDLF-synth aims to extend the capabilities of GaNDLF to include generative models, such as GANs, VAEs, and diffusion models, while adhering to the same principles.<br>
As an out-of-the-box solution, GaNDLF alleviates the need to build from scratch. Users may kickstart their project
by modifying only **a configuration (config) file** that provides guidelines for the envisioned pipeline
and **CSV inputs** that describe the training data.

## Range of GaNDLF-synth functionalities:

- Supports multiple
    - Deep Generative model architectures
    - Channels/modalities 
    - Labeling schemes (per patient, per custom class, unlabeled)
- Support of multiple loss, optimizers, scheduler, data augmentation, and evaluation metrics via interfacing GaNDLF
- Multi-GPU and multi-node training and inference support, integrating DistributedDataParallel [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [deepspeed](https://www.deepspeed.ai/); parallelism present both on the model and data level
- Leverages robust open source software - [Pytorch Lightning](https://lightning.ai/docs/pytorch/2.4.0/starter/introduction.html), [monai-generative](https://github.com/Project-MONAI/GenerativeModels)
- *Zero*-code needed to train robust models and generate synthetic data
- *Low*-code requirement for customization and addition of custom models and training logic
- [Automatic mixed precision](https://lightning.ai/docs/pytorch/2.4.0/common/precision_intermediate.html) support

## Table of Contents

- [Getting Started](./getting_started.md)
- [Application Setup](./setup.md)
- [Usage](./usage.md)
    - [Customize the training and inference](./customize.md)
- [Extending GaNDLF](./extending.md)
- [FAQ](./faq.md)
- [Acknowledgements](./acknowledgements.md)

## Citation
Please cite the following article for GaNDLF-synth:
<!-- TODO -->

## Contact
GaNDLF developers can be reached via the following ways:
<!-- TODO -->
<!-- - [GitHub Discussions](https://github.com/mlcommons/GaNDLF/discussions) -->
- [Email](mailto:gandlf@mlcommons.org)
