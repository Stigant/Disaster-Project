import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from joblib import dump

#Custom Classes
from OneByOneClassifier import OneByOneClassifier
from sklearn.multioutput import MultiOutputClassifier
from ThresholdClassifier import ThresholdClassifier
from IfThenClassifier import IfThenClassifier
from DefaultClassifier import DefaultClassifier

from misc import load_data, tokenize, keep_message, keep_genres, thresh_fun, no_entries_in, const


def model_features():
    """Fit transformer to extract features on df"""
    features=FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('keep_message', FunctionTransformer(keep_message)),
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                    ])),
                ('keep_others', FunctionTransformer(keep_genres))
                ])
    return features

def related_model():
    """SGDC classifier to predict if related, defaults to not related if
    no words in the message are recognised"""
    SGDC=SGDClassifier(n_jobs=-1)
    model=SGDC=DefaultClassifier(SGDC, {0:no_entries_in})
    return model

##F-beta scorer for GridSearch
fb_scorer = make_scorer(fbeta_score, beta=3/2)

def cat_model():
    """LogisticRegression model to predict message categories"""
    clf=ThresholdClassifier(LogisticRegression(max_iter=1000, solver='liblinear'), const(0.25))

    sample_pipeline = ImbalancedPipeline([
        ('resample', SMOTE()),
        ('clf',clf)
    ])

    #GridSearch
    param_grid={'clf__classifier__C' : np.logspace(-3, 3, 7),
            'clf__classifier__penalty' : ['l1', 'l2'],
            'clf__threshold_fun':  [thresh_fun, const(0.5), const(0.25)]}
    grid=GridSearchCV(sample_pipeline, param_grid, cv=5, scoring=fb_scorer, refit=True, n_jobs=-1)

    return OneByOneClassifier(grid)

def build_model():
    #Build Parts
    features=model_features()
    SGDC=related_model()
    clf=cat_model()

    #Build Model
    model = IfThenClassifier(features, SGDC, clf, repredict=True)
    return model

def evaluate_model(model, X_test, Y_test, partial=False):
    #Predict
    Y_preds=model.predict(X_test)

    #Evaluate
    if partial==True:
        report=classification_report(Y_test,Y_preds, target_names=Y_preds.columns, zero_division=0, output_dict=True)
        print(pd.DataFrame([report['related'],report['weighted avg']], index=['Related', 'Weighted Average']))
    else:
        print(classification_report(Y_test,Y_preds, target_names=Y_preds.columns, zero_division=0))
    return


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        dump(model, file)
    return


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, eval = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        if database_filepath.lower()== 'default':
            database_filepath="data/Disaster-Messages-Categories.db"
        if model_filepath.lower()== 'default':
            model_filepath="models/classifier.pkl"

        df= load_data(database_filepath)
        X=df[['message', 'genre_social', 'genre_news']].copy()
        Y=df.drop(['message', 'genre_social', 'genre_news', 'original', 'child_alone'],
            axis=1).astype(int)
        if eval.lower() in ['true', 'partial']:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        else:
            print('No Evaluation')
            X_train=X
            Y_train=Y

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        if eval.lower() == 'true':
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, partial=False)
        elif eval.lower() == 'partial':
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, partial=True)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. The third argument '\
              'should be True if you wish to evaluate model performance' \
              'or partial to return a summary'
              '\n\nExample: python train_classifier.py '\
              'data/DisasterResponse.db models/classifier.pkl True')


if __name__ == '__main__':
    main()
