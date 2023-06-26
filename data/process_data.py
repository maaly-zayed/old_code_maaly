# import necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    The function to load data .
    Parameters:
    messages_path: the file path to the messages CSV dataset.
    categories_path: the file path to the categories CSV dataset.
    """
  # load messages and categories csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
  # merge the two datasets
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """
    Function to clean data included in the DataFrame and transform categories part
    Parameters:
    model: trained model.
    df (DataFrame): Merged DataFrame
    OUTPUT
        (DataFrame) df: Cleaned DataFrame
    """

    # get the 36 categories
    categories = df.categories.str.split(";", expand=True)
    # get the first row of the categories dataframe 
    first_row = categories.loc[0]
    # extract a list of new column names for categories by getting the first (n-2) characters from a string
    category_colnames = first_row.apply(lambda i: i[:-2])
    # rename the categories
    categories.columns = category_colnames
    for category in categories:
        # set each value to be the last character of the string
        categories[category] =  categories[category].str[-1]
        # convert the data for each column  from string to numeric
        categories[category] = pd.to_numeric(categories[category])
    # drop the original categories column from the df
    df = df.drop(columns = 'categories') 
    # concatenate the original dataframe with the new categories dataframe
    df = df.join(categories)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
     

def save_data(df, database_filename):
    """
    The function to save cleaned data into sql database .
    Parameters:
    df: cleaned datafram.
    database_filename: name of database
    Output: sql database
    """
    # save a dataframe to sqlite database
    print('Save the {} datafram to {} database: '.format(df, database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)



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
