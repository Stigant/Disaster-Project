import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


def load_data(database_filepath):
    """Load pandas dataframe from SQL database"""
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_table('DisasterTable',engine).set_index('id')
    return df

def tokenize(text):
    """Clean and tokenize text, then lemmatize"""
    #Clean+Tokenize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text.lower()
    tokens=word_tokenize(text)

    #Lemmatize
    stopWords=stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopWords]
    return tokens

def keep_message(X):
    return  X['message']
def keep_genres(X):
    return X[['genre_social', 'genre_news']]
def thresh_fun(z):
    return max(min(0.5,2*z.mean()), 0.25)
def no_entries_in(X):
    return np.diff(X[:,:-2].indptr) == 0
