#!/bin/bash

PYTHON=$(which python3)
PIP_UPGRADE=${PIP_UPGRADE:-1}
VENV=${VENV:-venv}
FRESH_INSTALL=${FRESH_INSTALL:-1}
INSTALL_PRE_COMMIT=${INSTALL_PRE_COMMIT:-1}
INSTALL_CUPY=${INSTALL_CUPY:-0}
CUPY_VERSION=${CUPY_VERSION:-cupy}

function install()
{
  # activate environment
  source "$VENV"/bin/activate

  # upgrade pip and setuptools
  if [ "$PIP_UPGRADE" -ne 0 ]; then
    pip install --upgrade pip setuptools wheel
  fi

  # install package
  pip install -e .

  # install cupy
  if [ "$INSTALL_CUPY" -eq 1 ]; then
    pip install "$CUPY_VERSION" --no-cache
  fi

  # install development packages
  pip install -r requirements_dev.txt

  # install pre-commit
  if [ "$INSTALL_PRE_COMMIT" -eq 1 ]; then
    pre-commit install
  fi

  # deactivate environment
  deactivate
}


if [ "$FRESH_INSTALL" -eq 1 ]; then
  echo -e "Creating new environment..."
  rm -rf "$VENV"
  $PYTHON -m venv "$VENV"
fi


install || deactivate


echo -e ""
echo -e "Command to activate environment:"
echo -e "\t\$ source $VENV/bin/activate"
echo -e ""
echo -e "Command to deactivate environment:"
echo -e "\t\$ deactivate"
echo -e ""
