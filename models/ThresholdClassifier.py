import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator,ClassifierMixin):
    """Class to predict against custom threshold,
        base estimator needs predict_proba method"""

    def __init__(self, classifier, threshold_fun=(lambda: 0.5)):
        self.classifier=classifier
        self.threshold_fun=threshold_fun
        self.threshold=0.5

    def fit(self, X, y):
        #Fit Threshold
        threshold=self.threshold_fun(y)
        #Fit classifer
        self.classifier.fit(X,y)

    def predict_proba(self,X):
        return self.classifier.predict_proba(X)

    def predict(self,X):
        #Get probas
        clf=self.classifier
        predictions=pd.Series(clf.predict_proba(X)[:,1])

        #Compare to threshold
        t=self.threshold
        y_pred = (predictions > 0.5).astype(int)
        return y_pred
