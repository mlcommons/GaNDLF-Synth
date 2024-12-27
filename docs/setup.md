# Setup/Installation Instructions

## Prerequisites

- Python3 with a preference for [conda](https://conda.io), and python version `3.9` (higher versions *might* work, but they are **untested**).
- Knowledge of [managing Python environments](https://docs.python.org/3/tutorial/venv.html). The instructions below assume knowledge of the [conda management system](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

We are working on supporting containerized versions of GaNDLF-Synth, which will be available soon.

## Optional Requirements

- **GPU compute** (usually STRONGLY recommended for faster training):
    - Install appropriate drivers:
        - [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us)
        - [AMD](https://www.amd.com/en/support)
    - Compute toolkit appropriate for your hardware:
        - NVIDIA: [CUDA](https://developer.nvidia.com/cuda-download) and a compatible [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide
        - AMD: [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)
- Windows: [Microsoft Visual C++ 14.0 or greater](http://visualstudio.microsoft.com/visual-cpp-build-tools). This is required for PyTorch to work on Windows. If you are using conda, you can install it using the following command for your virtual environment: `conda install -c anaconda m2w64-toolchain`.

## Installation

### Install PyTorch 

GaNDLF-Synth primary computational foundation is built on PyTorch and PyTorch Lightning, and as such it supports all hardware types that PyTorch supports. Please install PyTorch for your hardware type before installing GaNDLF-Synth. 

The version to use needs to be analogous with the [GaNDLF version in the setup.py file](https://github.com/mlcommons/GaNDLF-Synth/blob/main/setup.py#L36). For example, for a requirement of `gandlf==0.1.1`, the [PyTorch requirement is 2.3.1](https://github.com/mlcommons/GaNDLF/blob/0.1.1/setup.py#L40).

See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for more details. 

First, instantiate your environment
```bash
(base) $> conda create -n venv_gandlf python=3.11 -y
(base) $> conda activate venv_gandlf
(venv_gandlf) $> ### subsequent commands go here
```

You may install PyTorch to be compatible with CUDA, ROCm, or CPU-only. An exhaustive list of PyTorch installations for the specific version compatible with GaNDLF can be found here: https://pytorch.org/get-started/previous-versions/#v231

Use one of the following installation commands provided under the "Install PyTorch" section of the PyTorch website. The following example is for installing PyTorch 2.3.1 with CUDA 12.1:
- CUDA 12.1
```bash
(venv_gandlf) $> pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
```

### Install from Package Managers

This option is recommended for most users, and allows for the quickest way to get started with GaNDLF.

```bash
(venv_gandlf) $> pip install gandlf-synth # this will give you the latest stable release
```
You can also use conda
```bash
(venv_gandlf) $> conda install -c conda-forge gandlf-synth -y
```
Or install directly from the GitHub repository
```bash
(venv_gandlf) $> git clone git@github.com:mlcommons/GaNDLF-Synth.git
(venv_gandlf) $> cd GaNDLF-Synth
(venv_gandlf) $> pip install .
```

If you are interested in running the latest version of GaNDLF-Synth, you can install the nightly build by running the following command:

```bash
(venv_gandlf) $> pip install --pre gandlf-synth
```

You can also use conda
```bash
(venv_gandlf) $> conda install -c conda-forge/label/gandlf_synth_dev -c conda-forge gandlf-synth -y
```

Test your installation:
```bash
(venv_gandlf) $> gandlf-synth verify-install
```