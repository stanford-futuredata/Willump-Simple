import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

base_directory = "tests/test_resources/product_resources/"


def product_train(y, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    model = LogisticRegression(solver='liblinear')
    model = model.fit(x, y)
    return model


def product_predict(model, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        return model.predict(x)


def product_predict_proba(model, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    else:
        return model.predict_proba(x)[:, 1]


def product_score(true_y, pred_y):
    return 1 - np.sqrt(mean_squared_error(true_y, pred_y))


def transform_data(data, vectorizer):
    return vectorizer.transform(data["title"])
