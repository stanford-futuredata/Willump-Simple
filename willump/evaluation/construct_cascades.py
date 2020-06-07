from typing import Mapping, Callable, MutableMapping, List

from willump.graph.willump_graph_node import WillumpGraphNode
import sklearn
import pandas as pd
import numpy as np


def construct_cascades(model_data: Mapping,
                       model_node: WillumpGraphNode,
                       train_function: Callable, predict_function: Callable,
                       predict_proba_function: Callable, score_function: Callable,
                       cascades_dict: MutableMapping) -> None:
    X, y = model_data["inputs"], model_data["params"]
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25, random_state=42)
    train_set_full_model = train_function(train_y, train_X)
    feature_groups: List[str] = model_node.input_names
    feature_costs: Mapping[str, float] = \
        {model_node.input_names[i]: model_node.input_nodes[i].cost for i in range(len(feature_groups))}


def calculate_feature_importances(train_set_full_model, valid_X, valid_y, predict_function, score_function,
                                  feature_groups):
    np.random.seed(42)
    base_preds = predict_function(train_set_full_model, valid_X)
    base_score = score_function(valid_y, base_preds)
    feature_importances: MutableMapping[str, float] = {}
    for i, (feature_group, valid_x) in enumerate(zip(feature_groups, valid_X)):
        valid_X_copy = valid_X.copy()
        shuffle_order = np.arange(valid_x.shape[0])
        np.random.shuffle(shuffle_order)
        valid_x_shuffled = valid_x[shuffle_order]
        valid_X_copy[i] = valid_x_shuffled
        shuffled_preds = predict_function(train_set_full_model, valid_X_copy)
        shuffled_score = score_function(valid_y, shuffled_preds)
        feature_importances[feature_group] = base_score - shuffled_score
    return feature_importances


def train_test_split(X, y, test_size, random_state):
    train_y, valid_y = sklearn.model_selection.train_test_split(y, test_size=test_size, random_state=random_state)
    train_X, valid_X = [], []
    for x in X:
        train_x, valid_x = sklearn.model_selection.train_test_split(x, test_size=test_size, random_state=random_state)
        train_X.append(train_x)
        valid_X.append(valid_x)
    return train_X, valid_X, train_y, valid_y
