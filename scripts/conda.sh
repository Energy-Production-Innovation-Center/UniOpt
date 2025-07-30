#!/usr/bin/env bash
# This script automatically manages the conda environment for this project.

env_name=$(basename "$PWD")

if ! conda info --envs | grep -Eq "\b$env_name\b"; then
    conda create -y -n $env_name --no-default-packages python=3.12
    conda run --live-stream -n $env_name pip install --upgrade -e .[dev]
fi

if [ $# -gt 0 ]; then
    conda run --live-stream -n $env_name $@
else
    conda run --live-stream -n $env_name pip install --upgrade -e .[dev]
fi
