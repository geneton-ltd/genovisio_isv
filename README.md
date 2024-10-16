# Genovisio ISV

[![Python version](https://img.shields.io/badge/python-3.12+-green.svg)](https://www.python.org/downloads/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

CNV annotation and pathogenicity prediction tool.

## Installation

In python3.12 you can simply use pip to install ISV:

```bash
pip install git+https://github.com/geneton-ltd/genovisio_isv.git
```

Without python3.12, you can install isv using mamba:

```bash
mamba env create -f conda_example.yaml
```

This gives you the following entrypoint:

- `isv-predict` - running ISV for only prediction of annotated CNV

## Running

First, you need annotated CNV region, for example `annotation.json`. Then to predict, run:

```shell
isv-predict annotation.json --output isv.json 2> log.err
```

## Development

Poetry is used to package the application. It is required to run `poetry build` and `poetry install` to recreate the `poetry.lock` containing frozen versions of dependencies.
