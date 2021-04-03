import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class DefaultClassifier(BaseEstimator,ClassifierMixin):
    """Class that outputs default predictions for some entries. Takes a dictionary of key,value pairs
    where value(X) returns true/false array and each row returning True is predicted key"""
    def __init__(self, classifier, defaults):
        self.classifier=classifier
        self.defaults=defaults

    def fit(self, X, y):
        self.classifier.fit(X,y)

    def predict(self, X):
        y=pd.Series(self.classifier.predict(X))
        for k,v in self.defaults.items():
            X_is_def = pd.Series(v(X), index=y.index)
            y[X_is_def]=k
        return y
