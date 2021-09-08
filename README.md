# Gradual (In)Compatibility of Fairness Criteria

This repository is the official implementation of the paper "Gradual (In)Compatibility of Fairness Criteria."

## Requirements

You should have Python (ideally [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)) installed. The code has been built and tested with Python 3.7.9 with Anaconda 4.10.3.

If you use ``conda``, you can create a virtual environment from the ``environment.yml`` file. The name of the environment will be ``gradual-compatibility``. Notice, however, that the package ``pyitlib`` (which is used to evaluate the information-theoretic terms) cannot be installed through ``conda``, so we have to additionally install it through ``pip``.

```setup
conda env create -f environment.yml
conda activate gradual-compatibility
pip install pyitlib
```

If you do not have conda installed, you may also install the required packages through ``pip``:

```setup
pip install -r requirements.txt
```

We recommend setting up a [virtual environment](https://docs.python.org/3/library/venv.html) for these dependencies.

## Data

The datasets can be found in the folder `/data` and are taken from [Friedler et al. (2019)](https://github.com/algofairness/fairness-comparison).

## Lambda selection

To select the lambda parameter for the L2 regularization of each dataset, run the notebook ``L2 - regularization.ipynb``.

## Regularization evaluation

To evaluate the different regularizers in the paper, run these commands:

```train
python evaluate_german.py
python evaluate_compas.py
python evaluate_adult.py
```

The results of these evalutions are saved in the folder `/evaluations`.

## Plots

To plot the figures shown in the paper ``Plots - direct and indirect effects.ipynb``. Edit the notebook to add or remove figures. In addition to being shown in the notebook, the figures are saved (in higher resolution) in the folder `/images`.
