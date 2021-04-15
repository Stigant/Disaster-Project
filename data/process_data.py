import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads and merges data from message and category datasets"""
    df_messages=pd.read_csv(messages_filepath)
    ser_categories=pd.read_csv(categories_filepath).set_index('id')

    #Dummy outcomes
    labels=[x.strip('-[01]')  for x in ser_categories.iloc[0,0].split(';')]
    category_dummies=ser_categories.categories.apply(
            lambda x: [y[-1] for y in x.split(';')]).apply(pd.Series).astype(int)
    category_dummies.columns=labels

    #Dummy genres
    df_messages=pd.get_dummies(df_messages, columns=['genre'], drop_first=True)

    df=df_messages.merge(category_dummies, left_on='id', right_index=True, how='inner')
    return df


def clean_data(df):
    """Cleans data - deals with dupes, NaN's and non-binary entries"""
    #Deal with duplicate IDs
    dupes=df[df.index.duplicated(keep=False)]
    union=dupes.groupby(level=0).max()
    df.loc[union.index]=union
    df.drop_duplicates(inplace=True)

    #Deal with untranslated messages
    untranslated = df.original.isna()
    df.loc[untranslated,'original']=df.loc[untranslated, 'message']

    #Deal with NaN's and #NAME?'s'
    df.dropna(how='any', inplace=True)
    df=df=df[df.message != "#NAME?"].copy()

    #Rescale dummy_category entries to 0/1
    df.iloc[:,3:]=df.iloc[:,3:].astype(int).astype(bool)

    return df


def save_data(df, database_filepath, tablename):
    """Saves to SQL database"""
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql(tablename, engine, index=False, if_exists='replace')
    pass


def main():

    if len(sys.argv) ==2 and sys.argv[1].lower()=='default':
        input = ('data/disaster_messages.csv',
        'data/disaster_categories.csv','data/Disaster-Messages-Categories.db',
        'DisasterTable')
    else:
        input= sys.argv[1:]

    if len(input) == 4:
        messages_filepath, categories_filepath, database_filepath, tablename = input

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {} \n TABLE: {}'.format(database_filepath, tablename))
        save_data(df, database_filepath, tablename)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument and a name for the table as the final '\
              'argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster-Messages-Categories.db DisasterTable')


if __name__ == '__main__':
    main()
