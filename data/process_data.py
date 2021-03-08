import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    df_messages=pd.read_csv(messages_filepath)
    ser_categories=pd.read_csv(categories_filepath).set_index('id')

    #Dummy outcomes
    labels=[x.strip('-[01]')  for x in ser_categories.iloc[0,0].split(';')]
    category_dummies=ser_categories.categories.apply(
            lambda x: [y[-1] for y in x.split(';')]).apply(pd.Series)
    category_dummies.columns=labels

    #Dummy genres
    df_messages=pd.get_dummies(df_messages, columns=['genre'], drop_first=True)

    df=df_messages.merge(category_dummies, left_on='id', right_index=True, how='inner')
    return df


def clean_data(df):

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

    #Rescale dummy_catogory booleans
    def rescale(x):
        if isinstance(x,int):
            if x == 0:
                return x
            else:
                return 1
        else:
            return x
    df.iloc[:,3:]=df.iloc[:,3:].applymap(rescale)

    return df


def save_data(df, database_filepath, tablename):
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql(tablename, engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, tablename = sys.argv[1:]

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
