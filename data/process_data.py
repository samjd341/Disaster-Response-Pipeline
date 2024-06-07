# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Function to load data to be used for disaster response pipeline

    Parameters:
    - messages_filepath - file path of messages data
    - categories_filepath - file path of categories data

    Returns: 
    - pandas df with merged messages and categories data
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Function to clean data to be used for disaster response pipeline
    Parameters: 
    - df - pandas data frame containing disaster response data, specifically messages and categories
    
    Returns: 
    - pandas dataframe containing cleaned data for disaster response pipeline
    """
    #split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    #convert category values to binary values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # data quality - remove values that are non-binary
        categories[column] = categories[column].apply(lambda x: np.nan if x not in [0,1] else x)
    


    #function to replace nan values with a binary value, with a probability based on the distribution of values in the column
    def replace_nan_with_random_binary(column):
        """
        Function to replace NaN values with random binary values based on the overall distribution of the data.
        
        Parameters:
        - column: pandas Series representing the column to process.
        
        Returns:
        - pandas Series with NaN values replaced with random binary values.
        """
            
        # Calculate the probability of each binary value based on the overall distribution of non-NaN values
        prob_0 = (column == 0).sum() / len(column.dropna())
        prob_1 = 1 - prob_0

        # Replace NaN values with random binary values based on the calculated probabilities
        random_binary_values = np.random.choice([0, 1], size=len(column), p=[prob_0, prob_1])
        processed_column = column.where(~column.isna(), random_binary_values)
        
        return processed_column
    
    # Apply the function to replace NaN values with random binary values in all columns of the DataFrame
    categories = categories.apply(replace_nan_with_random_binary, axis=0)
    
    #replace categories column in df with new category columns
    # drop the original categories column from `df`
    df = df.drop(columns = 'categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Function to save data to a SQLlite database

    Parameters:
    - df: A pandas dataframe
    - database_filename: Filename of a SQLlite database

    Returns:
    - df saved to SQLlite database 
    """
    #save data to a sql database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response_Table', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()