# GaNDLF-Synth

Presenting the **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for **Synth**esis (GaNDLF-Synth), a unified abstraction to train various synthesis algorithms in a zero/low code approach.

## Why use this?

- Supports multiple
  - Generative model architectures
  - Data dimensions (2D/3D)
  - Channels/images/sequences 
  - Label conditioning schemes
  - Domain modalities (i.e., Radiology Scans and Digitized Histopathology Tissue Sections)
  - Problem types (synthesis, reconstruction)
  - Multi-GPU and multi-node training
- Built-in 
  - Support for parallel HPC-based computing
  - Support for training check-pointing
  - Support for [Automatic mixed precision](./docs/usage.md#mixed-precision-training)
- Robust data augmentation and preprocessing (via interfacing [GaNDLF](
  https://docs.mlcommons.org/GaNDLF/))
- Leverages robust open source software
- No need to write any code to generate robust models

## Documentation

GaNDLF has extensive documentation and it is arranged in the following manner:

- [Home](https://mlcommons.github.io/GaNDLF-Synth/)
- [Installation](./docs/setup.md)
- [Usage](./docs/usage.md)
- [Extension](./docs/extending.md)
- [Frequently Asked Questions](./docs/FAQ.md)
- [Acknowledgements](./docs/acknowledgements.md)

## Contributing

Please see the [contributing guide](./CONTRIBUTING.md) for more information.

## Disclaimer
- The software has been designed for research purposes only and has neither been reviewed nor approved for clinical use by the Food and Drug Administration (FDA) or by any other federal/state agency.
- This code (excluding dependent libraries) is governed by [the Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) provided in the [LICENSE file](./LICENSE) unless otherwise specified.


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
