# Welcome to the AIBAS Project: Football Points Prediction.

## This AI project is based on the material of the course "M. Grum: Advanced AI-based Application Systems - Data Science and Business Analytics" conducted at the University of Potsdam under the supervision of Prof. Dr.-Ing. Marcus Grum. 

This is an AI project to predict the number of points a football team achieves based on the number of games played and the number of goals scored. The data used in this project is scraped from the ESPN football leagues' standings. The script for scraping the data is provided in this repo under `web-scrap.py`. more than 6000 records have been fetched online, data preprocessing has been applied to the data leading to 5000 clean records. The data was split into training and testing data with an 80:20 ratio.
This project conducted some experiments to compare the OLS regression model and the neural network model.

### Before Running the project you need to have:
1- `docker` installed and running.\
2- `python` and `pip` already installed.\
3- install the dependencies in `requirements.txt` using ```pip install -r requirements.txt```

### To run this project:
1- run `docker-compose up` to run the docker pipeline. This will create a new directory called `tmp` which contains the data required to make predictions.
2- run ```python code/model-activation/NN_activation.py```. This will make a prediction based on the activation data provided in `tmp`.

## Overview of the repo.
### The directory `code`
This directory contains the script used for this project including all docker files.
### The directory `data`
This directory contains the data files used in this project which include `raw` data and `processed` data.
The directory also contains `model-files` which are the Final saved model
### The directory `experiments`
This directory contains the results of each experiment conducted in this project while training the NN model. Each experiment contains a different visualization of predictions vs actual data and the performance measures on the training and testing data.




