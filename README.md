# model-training

![Badge](https://gist.githubusercontent.com/Remi-Lejeune/6ff1588ffc7e3f2e26de1428ea3bde64/raw/90431bb13596c3bc38edae2d06b2ab3856a81efc/badge.svg)


In this assignment we have transferred a small kaggle model to a professional development environment. We will be using the following tools:
- [DVC](https://dvc.org) for data version control and machine learning reproducibility.
- [Git](https://git-scm.com) for version control.
- [Poetry](https://python-poetry.org) for dependency management.

## 🛠️ Installation

Prerequisites:
- Poetry
- Python 3.11

Then, navigate to the main directory (model-training) and run:
```sh
# If the lock file is out of date
poetry lock --no-update

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## 📂 Data Retrieval and Pipeline Execution
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

### Public Sharing of the Model
The model trained is shared publicly if desired. On default this is not enabled. To enable this, set the third argument for the dvc `train` step in the `dvc.yaml` file to `true`. To have this working, you do need access to the s3 bucket and this can be requested from the authors.

The model currently saved however is openly available and can be downloaded from the s3 bucket. See how in `train.py`.

## 📊 Code Quality Metrics
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

## 🧪 Running Tests
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
## 📝 Notes on the project structure
- The project is structured in a way that the data is stored in the data/raw folder. This is the data that is used for training and testing. The data is not stored in the repository, but is stored in a dvc remote. The data is fetched from the remote using the dvc fetch command. The data is then processed and stored in the data/processed folder. The processed data is used for training and testing. 
- The Python source code is split up in the src folder. Every stage of the pipeline is stored in a separate file to keep the code clean and maintainable. The code is tested using the pytest framework. The tests are stored in the tests folder. The tests are split up in fast tests and manual tests. The fast tests are ran automatically in the CI pipeline and do not require dvc pull. The manual tests require all data and can be ran using the dvc pull command. The training tests require training on top of the data and can be ran using the dvc pull command.