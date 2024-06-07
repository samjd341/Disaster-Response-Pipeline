# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from xgboost import XGBClassifier
import pickle
import re
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
nltk.download(['wordnet', 'punkt','stopwords'])


def load_data(database_filepath):
    """
    Function to load data from a database for a disaster response pipeline
    
    Parameters: 
    - database_filepath: filepath of database which contains a table called Disaster_Response_Table containing disaster response data
    
    Returns: 
    - X: pandas dataframe containing feature data to predict disaster repsonse
    - Y: pandas dataframe containing target data to predict disaster response
    - category_names: names of categories in the disaster response data
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Disaster_Response_Table', engine)
    
    #split data into X (feature variables) and Y (target variables)
    X = df['message'].values
    Y = df.drop(columns=['message','original', 'genre', 'id'])
    
    #find the category names
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
    Function to tokenize text
    Parameters:
    - text: string variable
     
    Returns:
     -  a tokenized list of the text, which has been normalized, lemmatized and with stop words removed
     """
    # normalize case and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #tokenize text
    tokens = word_tokenize(text)
    
    
    #lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    #set up empty list for the clean tokens
    clean_tokens = []
    
    #lemmatize and strip the tokens and remove stop words
    for tok in tokens :
        clean_tok = lemmatizer.lemmatize(tok).strip()

        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Function to build a model for a NLP pipeline to predict disaster response with a multi output classifier
    
    Returns: 
    - NLP model"""
    pipeline= Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(XGBClassifier()))
        ])

    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3]
        }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate NLP model
    Parameters:
    - model: NLP model
    - X_test: X test data for model
    - Y_test: Y test data for model
    - category_names: names of categories in the data
     
    Returns:
     -  modelling metrics for NLP model
     """

    Y_pred = model.predict(X_test)
    
    for i, column_name in enumerate(Y_test.columns):
        # Extract the true labels and predictions for the current column
        Y_true_col = Y_test.iloc[:, i]
        Y_pred_col = Y_pred[:, i]
        
        # Report F1 score, precision, and recall for the current column
        report = classification_report(Y_true_col, Y_pred_col)
        print("Metrics for column '{}':\n{}".format(column_name, report))

    def modelling_metrics(Y_test, Y_pred):
        '''
        Returns the modelling metrics for a multi-class - multi-output ML model

        Parameters: 
            Y_test: True Y values for test data
            Y_pred: predicted Y values for test data

        Returns:
            Printed metrics of average accuracy, average preicison and average recall

        '''
        
        
        # Calculate accuracy for each class separately
        accuracies = [accuracy_score(Y_test.iloc[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]
        # Calculate precision for each class separately
        precisions = [precision_score(Y_test.iloc[:, i], Y_pred[:, i], average='micro'  ) for i in range(Y_test.shape[1])]
        # Calculate recall for each class separately
        recalls = [recall_score(Y_test.iloc[:, i], Y_pred[:, i], average='micro') for i in range(Y_test.shape[1])]

        # Calculate the average accuracy, precision, and recall across all classes
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)

        print("Average Accuracy:", avg_accuracy)
        print("Average Precision:", avg_precision)
        print("Average Recall:", avg_recall)
    
    return modelling_metrics(Y_test, Y_pred)



def save_model(model, model_filepath):
    """
    Function to save model to pkl file
    Parameters:
    - model: NLP model
    - model_filepath: file path for model to be saved to
     
    Returns:
     -  exported model as a pickle file
     """
    #export model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categorY_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        

        print('Training model...')
        model.fit(X_train, Y_train)
        

        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categorY_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()