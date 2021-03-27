import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin



class OneByOneClassifier(BaseEstimator,ClassifierMixin):
    """Class to fit against each output variable in a DataFrame separately"""

    def __init__(self, classifier, sampler):
        self.classifer=classifier
        self.sampler=sampler
        self.classifiers=None
        self.samplers=None
        self.columns=None

    def fit(self, X, y):
        self.columns=y.columns
        self.classifiers={col:clone(self.classifer) for col in self.columns}
        self.samplers={col:clone(self.sampler) for col in self.columns}

        for col in self.columns:
            y_col=y[col]
            classifier=self.classifiers[col]
            sampler=self.samplers[col]
            X_sample,y_col_sample=sampler.fit_sample(X,y_col)
            classifier.fit(X_sample,y_col_sample)

    def sample(self,X,y):
        samplers=self.samplers
        samples={col: samplers[col].sample(X,y) for col in self.columns}
        X_samples={col: samples[col][0] for col in self.columns}
        y_samples={col: samples[col][1] for col in self.columns}
        return X_samples, y_samples

    def predict(self,X):
        classifiers=self.classifiers
        predictions={col: classifiers[col].predict(X) for col in self.columns}
        return pd.DataFrame(predictions)

    def predict_proba(self,X):
        classifiers=self.classifiers
        predictions={col: classifiers[col].predict_proba(X)[:,1] for col in self.columns}
        return pd.DataFrame(predictions)

class ThresholdClassifier(BaseEstimator,ClassifierMixin):
    """Class to predict against custom threshold,
        base estimator needs predict_proba method"""

    def __init__(self, classifier, threshold_fun=(lambda: 0.5)):
        self.classifer=classifier
        self.threshold_fun=threshold_fun
        self.threshold=0.5

    def fit(self, X, y):
        #Fit Threshold
        threshold=self.threshold_fun(y)
        #Fit classifer
        clf=self.classifier
        clf.fit(X)

    def predict_proba(self,X):
        return self.classifier.predict_proba(X)

    def predict(self,X):
        #Get probas
        clf=self.classifiers
        predictions=pd.DataFrame(clf.predict_proba(X)[:,1])

        #Compare to threshold
        t=self.threshold
        y_pred = (predictions > 0.5).astype(int)
        return y_pred


class IfThenClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Class to predict against one binary variable,
        then predict again if first prediction==1""" 
    def __init__(self, transformer, clf1, clf2):
        clf.transformer=transformer
        self.clf1=clf1
        self.clf2=clf2
        self.length=1

    def fit(self, X, y):
        #Ensure indices match
        y.index=X.index

        #Fit length
        self.length=y.shape[1]

        #Fit transformer
        X_transformed=self.transformer.fit_transform(X)
        #Fit clf1
        y1=y.iloc[:,0]
        clf1=self.clf1
        clf1.fit(X,y1)

        #Fit clf2
        y2==(y[y1==1]).iloc[:,1:]
        X2=X[y1==1]
        clf2.fit(self.transformer.transform(X),y2)

    def predict(self,X):
        #Make output DataFrame
        y=pd.DataFrame(
            [[0]*self.length]*X.shape[1],
            index=X.index)
        #Predict y1
        X_transformed=self.transformer.fit(X)
        y1_pred=self.clf1.predict(X)
        y.iloc[:,0]=y1
        keepers=y.iloc[:,0]==1

        #If y1 predict y2
        X_true_tranformed=self.transformer.transform(X[keepers])
        y[keepers]=self.cl2.predict(X_true_tranformed)
        return y
