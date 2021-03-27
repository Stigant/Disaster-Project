import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class OneByOneClassifier(BaseEstimator,ClassifierMixin):
    """Class to fit against each output variable in a DataFrame separately"""
    def __init__(self, classifier):
        self.classifer=classifier
        self.classifiers=None
        self.columns=None

    def fit(self, X, y):
        self.columns=y.columns
        self.classifiers={col:clone(self.classifer) for col in self.columns}

        for col in self.columns:
            y_col=y[col]
            classifier=self.classifiers[col]
            classifier.fit(X, y_col)

    def predict(self,X):
        classifiers=self.classifiers
        predictions={col: classifiers[col].predict(X) for col in self.columns}
        return pd.DataFrame(predictions)

    def predict_proba(self,X):
        classifiers=self.classifiers
        predictions={col: classifiers[col].predict_proba(X)[:,1] for col in self.columns}
        return pd.DataFrame(predictions)
