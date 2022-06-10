# Real-time estimation of the effective reproduction number of COVID-19 from social contact surveys
This repository contains simulation code for analyzing social mixing patterns in Hungary.

## Install
This project is developed and tested with Python 3.8+. To install project dependencies, execute the following steps:
1) Install *virtualenv*: `pip3 install virtualenv`
2) Create a virtual environment (named *venv*) in the project directory: `python3 -m venv venv`
3) Activate *venv*: `source venv/bin/activate`
4) Install dependencies from `requirements.txt`: `pip install -r requirements.txt`

## Data
See description about corresponding data about
- contact matrices in `./contact_matrix/README.md`
- epidemic model parameters, `R_eff` estimations in `./data/README.md`

## Simulation
Code for simulating the epidemic model and calculating effective reproduction number is available in folder `./src`. 
Here you find 
- the data loading functionalities (`./src/dataloader.py`)
- the class for the epidemic model (`./src/model.py`)
- the methods for calculating the effective reproduction number (`./src/r0.py`)
- the applicable seasonality functions (`./src/seasonality.py`)
- the complete pipeline for the simulation, i.e. iterative solution of the epidemic model 
dynamically changing the contact matrix (`./src/simulation.py`)
In this folder you can find a simple script for running the simulation (`./src/main.py`)

## Notebooks
The repository contains notebooks for the aggregating raw data, exploring the underlying data and
generating figures for the article. See these notebooks in `./notebooks` and run Jupyter server in the activated
virtual enviroment (see section _Install_) using command `jupyter notebook`.

## Plots
Preliminarily created and notebook-generated plots are available in the folder `./plots`.

## Dashboard
An interactive dashboard is available for the project, where along with the effective reproduction number 
other components of the simulation are additionally shown. For running the dashboard run command 
`python3 <PROJECT_DIR>/dash/app.py` and open the popped-up link for the localhost webserver.
