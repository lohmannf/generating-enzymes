# generating-enzymes

## Installation

It is recommended to use a virtual environment in which you can install ```genzyme``` in editable mode
```
python -m venv prot_env
source prot_env/bin/activate
pip install -e .
```

### Special Requirements
Packages that have to be loaded from GitHub:
```
mutedpy
stpy
```

## Usage
A short tutorial on how to add your own datasets, train the models and generate new sequences can be found in ```notebooks/tutorial_model_training.ipynb```. Directions for running evaluations can be found in ```notebooks/tutorial_model_evaluation.ipynb```.

```scripts``` contains code for a few convenience utilities.

## Package Structure
| ```genzyme``` submodule | Purpose |
| ------- | --------|
| ```models``` | Contains a class for each model as well as a factory method for instantiating them based on identifiers |
| ```data```| Contains a loader class for each model that handles model-specific preprocessing and implements helper functions for data handling |
| ```evaluation```| Contains different metrics and methods for conducting a performance evaluation of models |

## Data 
This repository works with multiple datasets. 

The Imine Reductase Dataset ([Gantz et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.08.588565v1)) can be downloaded [here](https://github.com/Hollfelder-Lab/lrDMS-IRED/blob/main/data/srired_active_data.csv) and should be placed in ```data/IRed```. 

A simulated dataset with decreasing entropy with increasing sequence position can be found in ```data/scheduled_entropy```. 

To load the MID1 dataset, run ```scripts/preprocess_mid1.py``` after pointing ```raw_data_path``` to your local directory containing the raw data.

If you want to add your own data, follow the instructions in ```notebooks/tutorial_model_training.ipynb```

