# `ifs_physics_common`: Infrastructure code and convenient utilities to implement IFS physical parameterizations in Python

## Installation

We suggest installing the Python package `ifs_physics_common` within an isolated virtual environment:

```shell
# create virtual environment under `venv/`
$ python -m venv venv

# activate the virtual environment
$ . venv/bin/activate

# upgrade base packages
(venv) $ pip install --upgrade pip setuptools wheel
```

The package can be installed using the Python package manager [pip](https://pip.pypa.io/en/stable/):

```shell
# install ifs_physics_common along with the minimal set of requirements
(venv) $ pip install .

# create a fully reproducible development environment with an editable installation of cloudsc4py
(venv) $ pip install -r requirements-dev.txt
(venv) $ pip install -e .
```
