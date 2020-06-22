import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import scipy.sparse

base_directory = "tests/test_resources/toxic_resources/"


def toxic_train(y, X_list):
    X = scipy.sparse.hstack([*X_list], format="csr")
    model = LogisticRegression(C=0.1, solver='sag')
    model = model.fit(X, y)
    return model


def toxic_predict(model, X_list):
    X = scipy.sparse.hstack([*X_list], format="csr")
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        return model.predict(X)


def toxic_confidence(model, X_list):
    X = scipy.sparse.hstack([*X_list], format="csr")
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    else:
        return model.predict_proba(X)[:, 1]


def toxic_score(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)


def transform_data(data, vectorizer):
    return vectorizer.transform(data)
