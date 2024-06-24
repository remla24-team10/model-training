# REMLA project for group10

![Badge](https://gist.githubusercontent.com/Remi-Lejeune/6ff1588ffc7e3f2e26de1428ea3bde64/raw/90431bb13596c3bc38edae2d06b2ab3856a81efc/badge.svg)


In this assignment we will be transferring a small kaggle model to a professional development environment. We will be using the following tools:
- dvc for data version control and machine learning reproducibility.
- git for version control.
- poetry for dependency management.

# How to Run
## Prerequisites
- Poetry
- Python 3.11

## Installation
Navigate to the main directory (model-training) and run:
```sh
# If the lock file is out of date
poetry lock --no-update

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Data Retrieval and Pipeline Execution
Navigate to the remla-group10 folder and run:
```sh
dvc fetch
dvc pull  # May not work, use the workaround below if needed
dvc repro
```

If dvc pull does not work, manually fetch the data files and run dvc repro:
```sh
dvc fetch data/raw/train.txt
dvc fetch data/raw/test.txt
dvc fetch data/raw/val.txt
dvc repro
```

## Code Quality Metrics
Ensure code quality with `pylint`, `mypy`, `bandit`, and `pre-commit`.

### Install Mypy Stubs
Install mypy stubs for your dependencies:
```sh
mypy --install-types
```
### Using Pre-commit
The goal is to run Black, Pylint, Mypy, and Bandit with the configurations specified in pyproject.toml with one command. To do this, first install pre-commit:
```sh
pre-commit install
```
Then run the checks with:
```sh
pre-commit run --all-files
```
Pre-commit also runs automatically before every commit, which is what we want for this project.

## Running Tests
In the activated virtual environment, run:
```sh
pytest
```

**For specific tests:**
- Quick tests (run automatically in CI, no DVC pull required):
```sh
pytest -m fast
```

- Tests requiring all data (requires dvc pull or dvc repro):
```sh
pytest -s -m manual
```

- Tests requiring training (expected to take 30 minutes):
```sh
pytest -s -m training
```