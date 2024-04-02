# `ifs_physics_common`: Infrastructure code and convenient utilities to implement IFS physical parameterizations in Python

## Installation

The code is bundled as the installable package `ifs_physics_common`. We recommend installing the package in an isolated virtual environment:

```shell
# create a dedicated virtual environment
$ python -m venv venv

# activate the virtual environment
$ source venv/bin/activate

# upgrade basic packages
$ (venv) pip install --upgrade pip setuptools wheel

# install ifs_physics_common in editable mode
$ (venv) pip install -e .[<optional-dependencies>]
```

`<optional-dependencies>` can be one of the following strings, or a comma-separated list of them:

* `dev`: get a full-fledged development installation;
* `gpu`: enable GPU support by installing CuPy from source;
* `gpu-cuda11x`: enable GPU support for NVIDIA GPUs using CUDA 11.x;
* `gpu-cuda12x`: enable GPU support for NVIDIA GPUs using CUDA 12.x;
* `gpu-rocm`: enable GPU support for AMD GPUs using ROCm.
