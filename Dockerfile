# especify the base image
FROM python:3.11-slim

# install pipenv as our project uses it
RUN pip install pipenv

# create a directory in the container
WORKDIR /app

# copy the files to the container
COPY Pipfile Pipfile.lock ./

# install the dependencies. we need system and deploy to stop pipenv from creating a new virtual environment
RUN pipenv install --system --deploy  

# copy the files to the container
COPY predict.py xgboost_model0.1_6_0.1_0.8_0.8_15.bin ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
