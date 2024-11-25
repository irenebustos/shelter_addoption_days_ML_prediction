## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Prerequisites](#prerequisites)
4. [Steps](#steps-to-install)
5. [Deployment](#deployment)
6. [Usage](#usage)

## Introduction

### Goal
The goal of this project is to predict the time it will take for an animal to be adopted.

## Dataset
Intakes: Records of animal intakes at a shelter in Austin.
Outcomes: Records of the outcomes for those same animals at the shelter.
The dataset creation process is detailed in the file named time_shelter_dataset. Below is a summary:

The data has been merged between intakes and outcomes, with a series of adjustments applied during the process:
When there is more than one intake for the same animal, the outcome is attributed based on the outcome closest to the intake datetime.
If an intake occurs during another intake-outcome period for the same animal, it is removed, as it is considered a database error.
Only outcome_type = Adoption is retained for simplicity, as other types of outcomes are not relevant for this predictive exercise (more details can be found in the file).

Column Descriptions inside the dataset:
1. **animal_id**:A unique identifier for each animal in the shelter system.
2. **name**: The given name of the animal. Missing values (e.g., NaN) indicate the animal did not have a name when brought into the shelter.
3. **datetime_intake**: The date and time when the animal was taken into the shelter.
4. **found_location**: The location where the animal was found, often including an address or general area description.
5. **intake_type**: The method or reason for the animal's intake. Examples include "stray" (animal found roaming) or "owner surrender."
6. **intake_condition**: The condition of the animal at the time of intake. Typical values include "normal" or descriptors of health issues.
7. **animal_type**: The species of the animal, such as "cat" or "dog."
8. **sex_upon_intake**: The sex and sterilization status of the animal upon intake (e.g., "intact female" or "neutered male").
9. **age_upon_intake**: The approximate age of the animal when it was taken into the shelter.
10. **breed**: The breed or breed mix of the animal. This can include pure breeds (e.g., "chihuahua") or mixed breeds (e.g., "pit bull mix").
11. **color**: The coat color of the animal, which can include single or multiple colors (e.g., "brown tabby," "white/tan").
12. **datetime_outcome** :The date and time when the animal left the shelter through adoption.
13. The final status of the animal upon leaving the shelter, "Adoption" in all cases in our sample.

## Prerequisites
- `git`: For cloning the repository.
- `anaconda`(`conda`): For creating and managing the virtual environment.
- `docker`: deployment and testing.
- `AWS CLI`

## Steps to install

1. clone repository: https://github.com/irenebustos/shelter_addoption_days_ML_prediction.git
2. create conda environment with python 3.11
 ```bash
   $ conda create -n ml-shelter python=3.10
   ```
3. Activate the the virtual environment:

   ```bash
   $ conda activate ml-shelter
   ```
4. install required python packages: 
   ```bash
    $ pip install -r requirements.txt
   ```
5. create docker container using puthon 3.11-slim:
   ```bash
    docker run -it --rm --entrypoint=bash python:3.11-slim
   ```
6. Create docker image for this model:

   ```bash
    docker build -t zoomcamp_shelter .
   ```
7. run the docker container: 
   ```bash
    docker run -it --rm -p 9696:9696 zoomcamp_shelter
   ```
To stop the service, press `Ctrl+C` in the command prompt.

test