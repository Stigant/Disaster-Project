import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

class IfThenClassifier(BaseEstimator, ClassifierMixin):
    """Class to predict against one binary variable,
        then predict again (against new variables(s)) if first prediction==1.
        If repredict=True then inputs which are not assigned a
        class in the second are reclassified as not in the primary class.
        """
    def __init__(self, transformer, clf1, clf2, repredict=False):
        self.transformer=transformer
        self.clf1=clf1
        self.clf2=clf2
        self.rep=repredict
        self.columns=None

    def fit(self, X, y):
        #Ensure indices match
        y.index=X.index

        #Fit output columns
        self.columns=y.columns

        #Fit transformer
        X_transformed=self.transformer.fit_transform(X)
        #Fit clf1
        y1=y.iloc[:,0]
        self.clf1.fit(X_transformed,y1)

        #Fit clf2
        y2=(y[y1==1]).iloc[:,1:]
        X2=X[y1==1]
        self.clf2.fit(self.transformer.transform(X2),y2)

    def predict(self,X):
        #Make output DataFrame
        n=len(self.columns)
        m=X.shape[0]
        y_pred=pd.DataFrame(
            [[0]*n]*m,
            index=X.index, columns=self.columns)

        #Predict y1
        X_transformed=self.transformer.transform(X)
        y1_pred=self.clf1.predict(X_transformed)
        y1_pred.index=y_pred.index
        y_pred.iloc[:,0]=y1_pred
        keepers = (y_pred.iloc[:,0]==1)

        #If y1 predict y2
        if keepers.any():
            X_true_tranformed=self.transformer.transform(X[keepers])
            y2_pred=pd.DataFrame(self.clf2.predict(X_true_tranformed))
            y2_pred.index=X[keepers].index
            y2_pred.columns=y_pred.columns[1:]
            y_pred.loc[keepers,y2_pred.columns]=y2_pred
            if self.rep:
                y_pred.loc[keepers,'related']=y2_pred.any(axis=1).astype(int)
        return y_pred
