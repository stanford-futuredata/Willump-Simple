from typing import Mapping, Callable, MutableMapping, List

import numpy as np
import pandas as pd
import sklearn

from willump.graph.willump_graph_node import WillumpGraphNode


def construct_cascades(model_data: Mapping,
                       model_node: WillumpGraphNode,
                       train_function: Callable, predict_function: Callable,
                       confidence_function: Callable, score_function: Callable,
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
    pretty_print(feature_costs, "Cost")
    pretty_print(feature_importances, "Importance")
    total_feature_cost = sum(feature_costs.values())
    best_selected_feature_indices, selected_threshold, min_expected_cost = None, None, np.inf
    last_candidate_length = 0
    for cost_cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        selected_indices = select_features(feature_costs=feature_costs,
                                           feature_importances=feature_importances,
                                           cost_cutoff=cost_cutoff * total_feature_cost)
        if len(selected_indices) != last_candidate_length:
            last_candidate_length = len(selected_indices)
            selected_feature_cost = sum(feature_costs[feature_groups[i]] for i in selected_indices)
            threshold, fraction_approximated = calculate_feature_set_performance(train_X, train_y, valid_X, valid_y,
                                                                                 selected_indices,
                                                                                 train_function, predict_function,
                                                                                 confidence_function, score_function,
                                                                                 train_set_full_model)
            expected_cost = fraction_approximated * selected_feature_cost + \
                            (1 - fraction_approximated) * total_feature_cost
            print("Cutoff: %f Threshold: %f Expected Cost: %f" % (cost_cutoff, threshold, expected_cost))
            if expected_cost < min_expected_cost:
                best_selected_feature_indices = selected_indices
                selected_threshold = threshold
                min_expected_cost = expected_cost
    cascades_dict["selected_feature_indices"] = best_selected_feature_indices
    cascades_dict["cascade_threshold"] = selected_threshold
    cascades_dict["full_model"] = train_function(y, X)
    cascades_dict["approximate_model"] = train_function(y, [X[i] for i in best_selected_feature_indices])


def calculate_feature_importances(train_set_full_model, valid_X, valid_y, predict_function, score_function,
                                  feature_groups):
    np.random.seed(42)
    base_preds = predict_function(train_set_full_model, valid_X)
    base_score = score_function(valid_y, base_preds)
    feature_importances: MutableMapping[str, float] = {}
    for i, (feature_group, valid_x) in enumerate(zip(feature_groups, valid_X)):
        valid_X_copy = valid_X.copy()
        if isinstance(valid_x, pd.DataFrame):
            original_index = valid_x.index
            shuffle_order = valid_x.index.values.copy()
            np.random.shuffle(shuffle_order)
            valid_x_shuffled = valid_x.reindex(shuffle_order).sort_index().set_index(original_index)
        else:
            shuffle_order = np.arange(valid_x.shape[0])
            np.random.shuffle(shuffle_order)
            valid_x_shuffled = valid_x[shuffle_order]
        valid_X_copy[i] = valid_x_shuffled
        shuffled_preds = predict_function(train_set_full_model, valid_X_copy)
        shuffled_score = score_function(valid_y, shuffled_preds)
        feature_importances[feature_group] = base_score - shuffled_score
    return feature_importances


def select_features(feature_costs, feature_importances, cost_cutoff):
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
    return selected_indices


def calculate_feature_set_performance(train_X, train_y, valid_X, valid_y, selected_indices,
                                      train_function, predict_function,
                                      confidence_function, score_function,
                                      train_set_full_model,
                                      accuracy_threshold=0.001):
    full_model_preds = predict_function(train_set_full_model, valid_X)
    full_model_score = score_function(valid_y, full_model_preds)
    selected_train_X = [train_X[i] for i in selected_indices]
    selected_valid_X = [valid_X[i] for i in selected_indices]
    approximate_model = train_function(train_y, selected_train_X)
    approximate_confidences = confidence_function(approximate_model, selected_valid_X)
    approximate_preds = predict_function(approximate_model, selected_valid_X)
    best_threshold, best_frac = None, None
    for cascade_threshold in [1.0, 0.9, 0.8, 0.7, 0.6]:
        cascade_preds = full_model_preds.copy()
        num_approximated = 0
        for i in range(len(approximate_confidences)):
            if approximate_confidences[i] > cascade_threshold or approximate_confidences[i] < 1 - cascade_threshold:
                num_approximated += 1
                cascade_preds[i] = approximate_preds[i]
        combined_score = score_function(valid_y, cascade_preds)
        frac_approximated = num_approximated / len(cascade_preds)
        if combined_score > full_model_score - accuracy_threshold:
            best_threshold, best_frac = cascade_threshold, frac_approximated
    assert (best_threshold is not None and best_frac is not None)
    return best_threshold, best_frac


def train_test_split(X, y, test_size, random_state):
    train_y, valid_y = sklearn.model_selection.train_test_split(y, test_size=test_size, random_state=random_state)
    train_X, valid_X = [], []
    for x in X:
        train_x, valid_x = sklearn.model_selection.train_test_split(x, test_size=test_size, random_state=random_state)
        train_X.append(train_x)
        valid_X.append(valid_x)
    return train_X, valid_X, train_y, valid_y


def pretty_print(dictionary: Mapping, what: str) -> None:
    ranking = sorted(list(dictionary.keys()), key=lambda x: dictionary[x])
    for r in ranking:
        print("Feature: %s %s: %f" % (r, what, dictionary[r]))
