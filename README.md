### Disaster Response Pipeline Project

## Project Summary

This project creates a NLP pipeline for a Disaster Response. It takes in a dataset containing messages received during a disaster, with pre-populated classifications. The output is a model and front-end that classifies new messages in order to ensure the messages are sent to the correct teams.

## Requirements

* Python 3.11+ 
* NumPy
* Pandas
* Scikit-learn
* SQLAlchemy
* Nltk
* XGBoost
* Pickle
* Plotly
* Flask
* Joblib

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## File Descriptions
* data/disaster_messages.csv - csv containing disaster response messages
* data/disaster_categories.csv - csv containing categories of disaster response messages
* data/process_data.py - Python script to process input data (disaster_messages.csv and disaster_categories.csv) and output clean data to DisasterResponse.db
* data/DisasterResponse.db - SQLlite database containing cleaned disaster response data
* models/train_classifier.py - Python script to build an NLP model to classify disaster response messages
* models/classifier.pkl - NLP model to classify disaster response messages
* app/templates/go.html - template for Flask app
* app/templates/master.html - template for Flask app
* app/run.py - Python script to run Flask app to classify new disaster response messages

