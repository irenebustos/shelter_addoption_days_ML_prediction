
# Animal Adoption Days Prediction
![Alt text](image.png)

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prerequisites](#prerequisites)
4. [Steps to Install](#steps-to-install)
5. [Steps to Create the Docker Container and Run the Service](#steps-to-create-the-docker-container-and-run-the-service)
6. [Usage](#usage)

## Introduction
This project aims to predict the time (in days) that an animal will spend in the shelter before being adopted. The shelter has limited capacity, and it is crucial to determine whether an animal will be adopted within a short timeframe (approximately two weeks) or take longer. This helps prioritize efforts for animals facing adoption challenges.

## Dataset
The dataset used comes from real records of the **Austin Animal Shelter** in Texas. You can explore the source data on their official website:  
[City of Austin Animal Shelter](https://www.austintexas.gov/austin-animal-center).

The dataset consists of two main parts:
- **Intakes**: Records of animal intakes at the shelter. [Details](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm/about_data)
- **Outcomes**: Records of the outcomes for those same animals. [Details](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238/about_data)

### Dataset Summary
- Records: 2014–11/10/2024 (date downloaded).
- The dataset was created by merging `intakes` and `outcomes` with adjustments:
  1. For animals with multiple intakes, the outcome closest to the intake was used.
  2. Overlapping intake periods for the same animal were removed.
  3. Only `Adoption` outcomes were kept.

Detailed steps can be found in the `time_shelter_dataset` file.

### Dataset Columns
1. **`animal_id`**: Unique ID for each animal.
2. **`name`**: Name of the animal (NaN if unnamed).
3. **`datetime_intake`**: Intake date and time.
4. **`found_location`**: Location where the animal was found.
5. **`intake_type`**: Reason or method of intake (e.g., "stray").
6. **`intake_condition`**: Condition of the animal at intake.
7. **`animal_type`**: Species of the animal (e.g., "cat"). For this model dogs and cats were used. 
8. **`sex_upon_intake`**: Sex and sterilization status upon intake.
9. **`age_upon_intake`**: Approximate age at intake.
10. **`breed`**: Animal breed or mix.
11. **`color`**: Coat color(s).
12. **`datetime_outcome`**: Adoption date and time.
13. **`outcome_type`**: Final outcome status (always "Adoption").

---

## Prerequisites
- **`git`**: To clone the repository.
- **`Anaconda` (`conda`)**: For virtual environment management.
- **`Docker`**: For deployment and testing.
- **`AWS CLI`**: For cloud-based services (optional).

---

## Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/irenebustos/shelter_addoption_days_ML_prediction.git

2. create conda environment with python 3.11
 ```bash
   conda create -n ml-shelter python=3.10
   ```
3. Activate the the virtual environment:

   ```bash
   conda activate ml-shelter
   ```
4. install required python packages (go to the path where the repository is located  ¨cd .../shelter_addoption_days_ML_prediction) 
   ```bash
    pip install -r requirements.txt
   ```
4. Install the pip environment. for that pipenv is needed
pip install pipenv

5. pipenv install

6. activate the environment 
pipenv shell


## Steps to create the docker container and run the service
1. create docker container using puthon 3.11-slim:
   ```bash
    docker run -it --rm --entrypoint=bash python:3.11-slim
   ```
2. exit from the docker container using 

   ```bash
    exit
   ```
3. Create docker image for this model:

   ```bash
    docker build -t zoomcamp_shelter .
   ```
4. run the docker container: 
   ```bash
    docker run -it --rm -p 9696:9696 zoomcamp_shelter

5. open another tab in the terminal and run the example predict_test_ where there is an example of an animal entering the shelter (values can be modifiyed):

   ```bash
    python predict_test_.py
   ```
To stop the service, press `Ctrl+C` in the terminal.


## Using model in AWS
the jupyter notebook called ¨predict_test_AWS.ipynb¨is ready with the ULR of the service and can be used.
See here how to use it:

[![Watch the video demo](https://img.youtube.com/vi/yXJ7W0A7if4/0.jpg)](https://youtu.be/yXJ7W0A7if4)