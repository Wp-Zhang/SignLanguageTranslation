Sign Language Translation
==============================

Capstone Project of MSDS at NEU

Project Organization
------------

    ├── LICENSE
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   └── SLR                 <- Sign Language Recognition Dataset
    │       ├── processed           <- Preprocessed data that is ready for modeling.
    │       └── raw                 <- The original, immutable data dump.
    │
    ├── docs                    <- Documentation.
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    └── src                     <- Source code for use in this project.
        ├── __init__.py         <- Makes src a Python module
        │
        └── SLR                 <- Source code for Sign Language Recognition
            ├── data            <- Scripts to preprocess and load data
            ├── models          <- Model-related scripts, including model construction and training
            ├── visualization   <- Scripts to create exploratory and results-oriented visualizations
            └── run.py          <- Script to train and evaluate model

--------
## Sign Language Recognition

### Environment Setup

Install `ctcdecode`

1. `git clone https://github.com/parlance/ctcdecode.git`
2. `cd ctcdecode && pip install .`

### Data Preprocessing

1. Download **Phoenix-14** dataset and extract it to `data/SLR/raw/`, rename the data folder from `phoenix2014-release` to `phoenix2014`

2. In the project root folder, run cmd `python src/SLR/data/preprocess.py --process-image --multiprocessing`


### Download Model Weights

Download pre-trained model weights, see [here](https://github.com/ycmin95/VAC_CSLR). Put the downloaded model weights under `models/SLR/`


### Evaluation

Run cmd `python src/SLR/run.py --load-weights models/SLR/resnet18_vac_smkd_dev_19.80_epoch35_model.pt --phase test`

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
