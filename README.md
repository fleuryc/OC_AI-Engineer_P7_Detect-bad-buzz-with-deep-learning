[![Python application](https://github.com/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bb259c87a77f4beab13c48f4d5b59afe)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/bb259c87a77f4beab13c48f4d5b59afe)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/dashboard)

- [Air Paradis : Detect  bad buzz with deep learning](#air-paradis--detect--bad-buzz-with-deep-learning)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

* * *

# Air Paradis : Detect  bad buzz with deep learning

Repository of OpenClassrooms' [AI Engineer path](https://openclassrooms.com/fr/paths/188-ingenieur-ia), project #7

Goal : use Azure ML and NLP techniques (Gensim, Bert, Keras, ...), to perform sentiment analysis and prediction on tweets.

You can see the results here :

-   [Presentation](https://fleuryc.github.io/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/index.html)
-   [Blog Article : Comparing Azure Tools for Sentiment Analysis](https://github.com/fleuryc/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/blob/main/blog/blog-article.md "Comparing Azure Tools for Sentiment Analysis")
-   [Notebook : HTML page with interactive plots](https://fleuryc.github.io/OC_AI-Engineer_P7_Detect-bad-buzz-with-deep-learning/notebook.html "HTML page with interactive plots")

## Installation

### Prerequisites

-   [Python 3.9](https://www.python.org/downloads/)

### Virtual environment

```bash
# python -m venv env
# > or just :
make venv
source env/bin/activate
```

### Dependencies

```bash
# pip install kaggle jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests matplotlib seaborn plotly numpy
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

## Usage

### Run Notebook

```bash
jupyter-lab notebooks/main.ipynb
```

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

-   Fix Plotly issues with JupyterLab

cf. [Plotly troubleshooting](https://plotly.com/python/troubleshooting/#jupyterlab-problems)

```bash
jupyter labextension install jupyterlab-plotly
```

-   If using Jupyter Notebook instead of JupyterLab, uncomment the following lines in the notebook

```python
import plotly.io as pio
pio.renderers.default='notebook'
```
