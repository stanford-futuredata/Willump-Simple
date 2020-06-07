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
    feature_importances = calculate_feature_importances(train_set_full_model=train_set_full_model,
                                                        valid_X=valid_X, valid_y=valid_y,
                                                        predict_function=predict_function,
                                                        score_function=score_function,
                                                        feature_groups=feature_groups)


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


def select_best_features(feature_groups, feature_costs, feature_importances, cost_cutoff):
    def knapsack_dp(values, weights, capacity):
        # Credit: https://gist.github.com/KaiyangZhou/71a473b1561e0ea64f97d0132fe07736
        n_items = len(values)
        table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
        keep = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)

        for i in range(1, n_items + 1):
            for w in range(0, capacity + 1):
                wi = weights[i - 1]  # weight of current item
                vi = values[i - 1]  # value of current item
                if (wi <= w) and (vi + table[i - 1, w - wi] > table[i - 1, w]):
                    table[i, w] = vi + table[i - 1, w - wi]
                    keep[i, w] = 1
                else:
                    table[i, w] = table[i - 1, w]
        picks = []
        K = capacity
        for i in range(n_items, 0, -1):
            if keep[i, K] == 1:
                picks.append(i)
                K -= weights[i - 1]
        picks.sort()
        picks = [x - 1 for x in picks]  # change to 0-index
        return picks
    scaled_total_cost = 1000
    total_cost = sum(feature_costs.values())
    scale_factor = scaled_total_cost / total_cost
    scaled_feature_costs = [round(c * scale_factor) for c in feature_costs.values()]
    scaled_cost_cutoff = round(cost_cutoff * scale_factor)
    selected_indices = knapsack_dp(list(feature_importances.values()), scaled_feature_costs, scaled_cost_cutoff)
    return [feature_groups[i] for i in selected_indices]


def train_test_split(X, y, test_size, random_state):
    train_y, valid_y = sklearn.model_selection.train_test_split(y, test_size=test_size, random_state=random_state)
    train_X, valid_X = [], []
    for x in X:
        train_x, valid_x = sklearn.model_selection.train_test_split(x, test_size=test_size, random_state=random_state)
        train_X.append(train_x)
        valid_X.append(valid_x)
    return train_X, valid_X, train_y, valid_y
