# REMLA project for group10

![Badge](https://gist.githubusercontent.com/Remi-Lejeune/6ff1588ffc7e3f2e26de1428ea3bde64/raw/90431bb13596c3bc38edae2d06b2ab3856a81efc/badge.svg)


In this assignment we will be transferring a small kaggle model to a professional development environment. We will be using the following tools:
- dvc for data version control and machine learning reproducibility.
- git for version control.
- poetry for dependency management.

### How to run
To run this code you need to have poetry and python3.11 installed. 
You can install the packages by running the following commands:
(this should be executed in the phishing-detection folder)

- if the lock file is out of date:
- ```poetry lock --no-update```

- ```poetry install```
- ```poetry shell```

To retrieve the data and run the pipeline:
(this should be executed in the remla-group10 folder)
- ```dvc fetch```
- ```dvc pull``` (may not work, in which case use the workaround below)
- ```dvc repro```

In case dvc pull does not work, fetch the 3 data files manually and run dvc repro:
- ```dvc fetch data/raw/train.txt```
- ```dvc fetch data/raw/test.txt```
- ```dvc fetch data/raw/val.txt```
- ```dvc repro```

To run the code quality metrics:
(this should be executed in the phishing-detection folder)
- ```pylint ./src ./tests```
- ```bandit ./ -r```

The project will be restructured in the future such that there is a single root folder from which all scripts can be executed from.

### How to run the tests 
After entering the virtual environment run: ```pytest```.
- ```pytest -m fast``` for quick tests, these are ran automatically in the CI pipeline and do not require dvc pull.
- ```pytest -s -m manual``` for tests that require all data which can be downloaded through dvc pull or dvc repro.
- ```pytest -s -m training``` for tests that require training on top of dvc pull or dvc repro, this is expected to take 30 minutes.