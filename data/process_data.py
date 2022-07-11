import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads two datasets and merge them together

    Input: Messages and Categories dataset path

    Output: Merged dataset
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id", how="outer")
    return df


def clean_data(df):
    '''
    Cleans dataset by:
    - Splitting categories column into multiple columns
    - Updating columns on the dataset created above
    - Converting string values into Boolean by picking the last character
    - Replacing original column with the newly created dataset
    - Check and drop duplicate rows

    Input: Merged messages and categories dataset

    Output: Cleaned dataset
    '''
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : int(x[-1]))
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Saves dataset into a SQL db

    Input: Dataset and database filename

    Output: None
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_df', engine, index=False)
    pass  


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