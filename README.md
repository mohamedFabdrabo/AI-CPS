# Welcome to the AIBAS Project: Football Points Prediction.

## This AI project is based on the material of the course "M. Grum: Advanced AI-based Application Systems - Data Science and Business Analytics" conducted at the University of Potsdam under the supervision of Prof. Dr.-Ing. Marcus Grum. 

This is an AI project to predict the number of points a football team achieves based on the number of games played and the number of goals scored. The data used in this project is scraped from the ESPN football leagues' standings. The script for scraping the data is provided in this repo under `web-scrap.py`. more than 6000 records have been fetched online, data preprocessing has been applied to the data leading to 5000 clean records. The data was split into training and testing data with an 80:20 ratio.
This project conducted some experiments to compare the OLS regression model and the neural network model.

## Overview of the repo.
### The directory `code`
This directory contains the script used for this project including all docker files.
### The directory `data`
This directory contains the data files used in this project which include `raw` data and `processed` data.
The directory also contains `model-files` which are the Final saved model
### The directory `experiments`
This directory contains the results of each experiment conducted in this project while training the NN model. Each experiment contains a different visualization of predictions vs actual data and the performance measures on the training and testing data.


### To run the NN model Activation directly with Docker-compose:
Run 
```
docker-compose -f docker-compose.yml -f docker-compose.nn.yml up
```
This will run the docker pipeline, docker will fetch the three main images pushed on docker hub. Install the requirements and make predictions using the NN model in the `tmp/knowledgeBase` folder and the activation data inside `tmp/activationBase` 

### To run the OLS model Activation directly with Docker-compose:
Run 
```
docker-compose -f docker-compose.yml -f docker-compose.ols.yml up
```
This will activate the docker ols pipeline and do the same as the NN pipeline except running using ols saved model.

### To run the activation script without docker:
1. you must have `docker` installed and running.\
2. also, `python` and `pip` already installed.\
3. install the dependencies in `requirements.txt` using ```pip install -r requirements.txt```
4. run `docker-compose up` to run the docker pipeline. This will create a new directory called `tmp` which contains the data required to make predictions.
6. run ```python code/model-activation/NN_activation.py```. This will make a prediction based on the activation data provided in `tmp`.

## important commands for the project:
### project initialization:
- In order to scrap the online data again run: `python code/web-scrapping/web-scrap.py`
- The above script will scrap the online data from ESPN and load the data to `data/raw/joint_data_collection.csv`
- To split the raw scraped data into training and testing sets run: `python code/data-prep/data-preprocessing/data_preprocessing.py`
- The script will split the data with 80:20 ratio and store the data in `data/processed`
- to retrain the NN model with the processed data: `python code/model-training/NN_train_test.py`
- to retrain the OLS model with the processed data: `python code/model-training/ols_train_test.py`

### Docker Build Utilization:
#### learning_base docker image:
- Build the container: `docker build --tag <DOCKER_USERNAME>/learningbase_points_prediction:latest -f code/docker/learningBase_Dockerfile .`
- Test the container : `docker run -it --rm <DOCKER_USERNAME>/learningbase_points_prediction sh`
- Push on Dockerhub  : `docker image push <DOCKER_USERNAME>/learningbase_points_prediction`

#### activation_base docker image:
- Build the container: 
`docker build --tag <DOCKER_USERNAME>/activationbase_points_prediction:latest -f code/docker/activationBase_Dockerfile .`
- Test the container : `docker run -it --rm <DOCKER_USERNAME>/activationbase_points_prediction:latest sh`
- Push on Dockerhub  : `docker image push <DOCKER_USERNAME>/activationbase_points_prediction:latest`

#### knowledge_base docker image:
- Build the container: 
`docker build --tag <DOCKER_USERNAME>/knowledgebase_points_prediction:latest -f code/docker/knowledgeBase_Dockerfile .`
- Test the container : `docker run -it --rm <DOCKER_USERNAME>/knowledgebase_points_prediction:latest sh`
- Push on Dockerhub  : `docker image push <DOCKER_USERNAME>/knowledgebase_points_prediction:latest`

#### code_base docker image (not pushed):
- Build the container: 
`docker build --tag <DOCKER_USERNAME>/codebase_points_prediction:latest -f code/docker/codeBase_Dockerfile .`
- Test the container : `docker run -it --rm <DOCKER_USERNAME>/codebase_points_prediction:latest sh`


