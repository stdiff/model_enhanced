from unittest import TestCase

from model_enhanced import *

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression


class ModelTest(TestCase):
    def test_with_preprocessor(self):
        data = load_iris()

        df = pd.DataFrame(data.data, columns=["v%d" % j for j in range(1, 5)])
        df.index = [i**2 + 1 for i in range(df.shape[0])]
        df["Species"] = pd.Series([data["target_names"][i] for i in data["target"]],
                                  index=df.index)

        target = "v1"
        X = df.drop(target, axis=1)
        y = df[target]

        lb = LabelBinarizer()
        lb.fit(df["Species"])

        def preprocessor(X:pd.DataFrame) -> pd.DataFrame:
            """
            Note that this function requires a fitted LabelBinarizer
            """
            X_processed = X.drop("Species", axis=1)
            X_pivot = pd.DataFrame(lb.transform(df["Species"]),
                                   columns=lb.classes_,
                                   index=X.index)
            return pd.concat([X_processed, X_pivot], axis=1)

        lr = LinearRegression()
        lr.fit(preprocessor(X),y)

        my_model = RegressionModel(preprocessor,lr)
        y_hat = my_model.predict(X)

        self.assertTrue(isinstance(y_hat,pd.Series))
        self.assertTrue(list(X.index), list(y_hat.index))


    def test_ClassificationModel(self):
        data = load_iris()

        X = pd.DataFrame(data.data, columns=["v%d" % j for j in range(1, 5)])
        X.index = [i**2 for i in range(X.shape[0])]
        y = pd.Series([data["target_names"][i] for i in data["target"]],
                      name="Species", index=X.index)

        def preprocessor(X:pd.DataFrame) -> pd.DataFrame:
            """
            Remove just one column from the DataFrame
            """
            return X.drop("v1", axis=1)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("plr", LogisticRegression(C=1.0, random_state=3))
        ])

        pipeline.fit(preprocessor(X),y)

        my_model = ClassificationModel(preprocessor, pipeline)

        y_hat = my_model.predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))
        self.assertEqual(y_hat.dtype, "object")
        self.assertEqual(list(X.index), list(y_hat.index))

        Y_proba = my_model.predict_proba(X)
        self.assertTrue(isinstance(Y_proba, pd.DataFrame))
        self.assertEqual(Y_proba.shape[1], 3)
        self.assertEqual(list(X.index), list(Y_proba.index))


    def test_BinaryModel(self):
        data = load_breast_cancer()

        columns = [c.replace(" ","_") for c in data.feature_names]
        X = pd.DataFrame(data.data, columns=columns)
        X.index = [i**2 + 1 for i in range(X.shape[0])]
        y = pd.Series([data["target_names"][i] for i in data["target"]],
                      name="label", index=X.index)

        pos_label = "malignant"
        neg_label = "benign"

        def preprocessing(X:pd.DataFrame) -> pd.DataFrame:
            return X.drop("mean_texture", axis=1)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("plr", LogisticRegression(C=1.0, random_state=4))
        ])

        pipeline.fit(preprocessing(X), y)

        threshod = 0.4
        my_model = BinaryModel(processor=preprocessing,
                               model=pipeline,
                               threshold=threshod,
                               pos_label=pos_label)

        y_pred = my_model.predict(X)
        self.assertTrue(isinstance(y_pred, pd.Series))

        Y_proba = my_model.predict_proba(X)
        self.assertTrue(isinstance(Y_proba, pd.DataFrame))

        y_proba = my_model.compute_score(X)
        y_hat = y_proba.apply(lambda y: 1 if y >= threshod else 0)
        self.assertTrue(isinstance(y_proba, pd.Series))
        self.assertEqual((y_pred == y_hat).mean(), 1)

