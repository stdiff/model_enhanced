from typing import Union,Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class BaseModel:
    def __init__(self,
                 processor: Callable[[pd.DataFrame], pd.DataFrame],
                 model: Union[BaseEstimator,Pipeline]):
        """
        :param processor: function converts a DataFrame to a DataFrame
        :param model: scikit-learn classifier/pipeline
        """
        self.processor = processor
        self.model = model

    def predict(self, X:pd.DataFrame) -> pd.Series:
        """
        Apply the model to the given DataFrame

        :param X: DataFrame
        :return: Series of predictions (with the same index as X)
        """
        X_processed = self.processor(X)
        y_pred = self.model.predict(X_processed)
        return pd.Series(y_pred, name="prediction", index=X.index)


class ClassificationModel(BaseModel):
    def predict_proba(self, X:pd.DataFrame) -> pd.DataFrame:
        X_processed = self.processor(X)
        Y_proba = self.model.predict_proba(X_processed)
        return pd.DataFrame(Y_proba, columns=self.model.classes_, index=X.index)


class BinaryModel(ClassificationModel):
    def __init__(self,
                 processor: Callable[[pd.DataFrame], pd.DataFrame],
                 model: Union[BaseEstimator,Pipeline],
                 threshold: float,
                 pos_label: Union[str,int]=1):
        """
        This class provides API for a binary classifier with a threshold.
        If you want to apply an classifier in an ordinary way, you should
        use ClassificationModel.

        :param processor: function converts a DataFrame to a DataFrame
        :param model: scikit-learn classifier/pipeline
        :param threshold: threshold (if probability >= threshold, then it is a positive class)
        :param pos_label: label of the positive class of the target variable
        """
        super().__init__(processor, model)
        self.threshold = threshold
        self.pos_label = pos_label

    def compute_score(self, X:pd.DataFrame) -> pd.Series:
        """
        Compute the probability of the positive class

        :param X: DataFrame
        :return: Series of probabilities
        """
        return self.predict_proba(X)[self.pos_label]


        #col = list(self.model.classes_).index(self.pos_label)
        #X_processed = self
        #return pd.Series(self.model.predict_proba(X)[:, col], name="score", index=X.index)


    def predict(self, X:pd.DataFrame) -> pd.Series:
        """
        Note that this methods produces a Series with 0 and 1.
        Namely not with the original labels.

        :param X: DataFrame
        :return: Series of 1 and 0.
        """
        y_score = self.compute_score(X)
        return y_score.apply(lambda y: 1 if y >= self.threshold else 0)


class RegressionModel(BaseModel):
    pass

