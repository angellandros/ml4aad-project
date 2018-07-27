# ML4AAD Final Project
Hyperparameter Optimization using Bayesian Optimization with Random Forest Regression

## Installation
To install the requirements, please use the file `requirements.txt`:
```bash
$ pip install -r requirements.txt
```

## Usage
The main executable file `runner.py` accepts no parameters, instead one can change the parameters in `scenario.txt`. To run the script, simply run `runner.py`:
```bash
$ python runner.py
```
The script will find a well-performing hyperparameter setting, evaluate it, and then generate a plot comparing it to the default parameters.

## Results
The BO algorithm reduces PAR10 to half, and PAR1 for 25%. The timouts are also half. Please refer to the figure `fig-bo.png`.
