import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
import dill

#Custom Classes
from OneByOneClassifier import OneByOneClassifier
from ThresholdClassifier import ThresholdClassifier
from IfThenClassifier import IfThenClassifier

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

def model_features():
    """Fit transformer to extract features on df"""
    features=FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('keep_message', FunctionTransformer(lambda X: X.message)),
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                    ])),
                ('keep_others', FunctionTransformer(lambda X: X[['genre_social', 'genre_news']]))])
    return features

def related_model():
    """SGDC classifier to predict if related"""
    SGDC=SGDClassifier(n_jobs=-1)
    return SGDC

def cat_model():
    """LogisticRegression model to predict message categories"""
    thresh_fun= lambda z: max(min(0.5,2*z.mean()), 0.25)
    clf=ThresholdClassifier(LogisticRegression(solver='newton-cg'), thresh_fun)

    sample_pipeline = ImbalancedPipeline([
        ('resample', SMOTE()),
        ('classifier',clf)
    ])

    clf=OneByOneClassifier(sample_pipeline)
    return clf

def build_model():
    #Build Parts
    features=model_features()
    SGDC=related_model()
    clf=cat_model()

    #Build Model
    model = IfThenClassifier(features, SGDC, clf)
    return model

def evaluate_model(model, X_test, Y_test):
    #Predict
    Y_preds=model.predict(X_test)

    #Evaluate
    report=classification_report(Y_test,Y_preds, target_names=Y_preds.columns, zero_division=0, output_dict=True)
    print(pd.DataFrame([report['related'],report['weighted avg']], index=['Related', 'Weighted Average']))
    return


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        dill.dump(model, file)
    return


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, eval = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        if database_filepath.lower()== 'default':
            database_filepath="../data/Disaster-Messages-Categories.db"
        if model_filepath.lower()== 'default':
            model_filepath="classifier.pkl"

        df= load_data(database_filepath)
        X=df[['message', 'genre_social', 'genre_news']].copy()
        Y=df.drop(['message', 'genre_social', 'genre_news', 'original', 'child_alone'],
            axis=1).astype(int)
        if eval.lower() == 'true':
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        else:
            X_train=X
            Y_train=Y

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        if eval.lower() == 'true':
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test)

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
