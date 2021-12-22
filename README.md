[![Python application](https://github.com/fleuryc/Template-Python/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/Template-Python/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/Template-Python/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/Template-Python/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/b03fbc514ea44fce83fe471896566cfd)](https://www.codacy.com/gh/fleuryc/Template-Python/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/b03fbc514ea44fce83fe471896566cfd)](https://www.codacy.com/gh/fleuryc/Template-Python/dashboard)

- [Project](#project)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Run](#run)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

* * *

# Project

-   What ?
-   Why ?
-   How ?

## Installation

### Prerequisites

-   environment variables
-   external programs
-   compatible versions

### Virtual environment

```bash
make venv
source env/bin/activate
```

### Dependencies

```bash
# pip install jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests matplotlib seaborn plotly numpy
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

## Usage

### Run

-   execution instructions

### Quality Assurance

```bash
# make isort
# make format
# make lint
# make bandit
# make mypy
# make test
# > or just :
make qa
```

## Troubleshooting

-   known issues
-   how to fix them
