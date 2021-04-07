import json
import plotly
import joblib
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

import sys

sys.path.append("../models")

app = Flask(__name__)






#Redefine misc functions
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


#Load data+model
df = load_data("../data/Disaster-Messages-Categories.db")
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/templates/master.html')
def index():
    # extract data needed for visuals
    #Graph 1
    genres = keep_genres(df)
    genres.loc[:,'genre_other']=~(genres.any(axis=1))

    genre_counts=genres.sum().transpose().copy()
    genre_names = [x.split('_')[1] for x in genre_counts.index]
    genre_counts=genre_counts.tolist()

    related_counts= list(df.pivot_table(columns=['genre_social','genre_news'],
                    values='related', aggfunc='sum').values[0][::-1])


    #Graph 2

    cat_counts=df.iloc[:,4:].sum().sort_values(ascending=False)
    cats=list(cat_counts.index)
    cat_counts=cat_counts.values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name='Recieved Messages'
                ),
                Bar(
                x=genre_names,
                y=related_counts,
                name='Relevant Messages'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cats,
                    y=cat_counts,
                    name='Categories')
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'margin': {
                    'l': 50,
                    'r': 50,
                    'b': 100,
                    't': 100,
                    'pad': 4
                },

                'yaxis': {
                    'title': {
                        'text': "Count",
                        'standoff': 0
                        }
                },


                'xaxis': {
                    'title': {
                        'text': "Message Category",
                        'standoff': 100
                        },
                    'tickangle': 30
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, genre_data=(genre_names, genre_counts))


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    cols=['message', 'genre_social', 'genre_news']

    message = request.args.get('query', '')
    input=pd.DataFrame([[message, 1, 0]], columns=cols)

    # use model to predict classification for query
    prediction=model.predict(input)
    classification_labels = prediction.values[0]
    classification_results = dict(zip(prediction.columns, classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=message,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
