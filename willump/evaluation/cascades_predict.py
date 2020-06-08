import ast
import copy
import inspect
from typing import Mapping, Callable

import numpy as np
import pandas as pd
import scipy

from willump.graph.utilities import create_function_ast, create_model_ast
from willump.graph.willump_graph_node import WillumpGraphNode


def predict_cascades(func: Callable, model_node: WillumpGraphNode,
                     predict_function: Callable, predict_proba_function: Callable,
                     cascades_dict: Mapping) -> Callable:
    class WillumpCascadesPredict(ast.NodeTransformer):
        def visit_FunctionDef(self, orig_ast: ast.FunctionDef) -> ast.AST:
            cascades_body = []
            # Compute selected features.
            for node in selected_feature_nodes:
                cascades_body += node.get_ast()
            # Predict with approximate model.
            approximate_model_ast = create_model_ast(function_name=predict_proba_function.__name__,
                                                     output_name="__willump_approximate_preds",
                                                     input_names=selected_feature_names,
                                                     model_param="__willump_approximate_model")
            cascades_body += approximate_model_ast
            # Get indices that can't be approximated.
            unapproximated_indices_ast = create_function_ast(function_name="__willump_get_unapproximated_indices",
                                                             output_name="__willump_unapproximated_indices",
                                                             input_names=["__willump_approximate_preds",
                                                                          "__willump_cascade_threshold"])
            cascades_body += unapproximated_indices_ast
            # Only compute remaining features for unapproximated indices.
            shortened_inputs = set()
            for node in remaining_feature_nodes:
                for input_name in node.input_names:
                    if input_name not in shortened_inputs:
                        shorten_ast = create_function_ast(function_name="__willump_select_unapproximated_rows",
                                                          output_name=input_name,
                                                          input_names=[input_name, "__willump_unapproximated_indices"])
                        cascades_body += shorten_ast
                        shortened_inputs.add(input_name)
                cascades_body += node.get_ast()
            # Shorten the selected features.
            for name in selected_feature_names:
                shorten_ast = create_function_ast(function_name="__willump_select_unapproximated_rows",
                                                  output_name=name,
                                                  input_names=[name, "__willump_unapproximated_indices"])
                cascades_body += shorten_ast
            # Predict with full model.
            full_model_ast = create_model_ast(function_name=predict_function.__name__,
                                              output_name="__willump_full_preds",
                                              input_names=model_node.input_names,
                                              model_param="__willump_full_model")
            cascades_body += full_model_ast
            # Return combined predictions.
            combine_predictions_ast = create_function_ast(function_name="__willump_combine_predictions",
                                                          output_name="__willump_final_predictions",
                                                          input_names=["__willump_approximate_preds",
                                                                       "__willump_full_preds",
                                                                       "__willump_cascade_threshold"])
            cascades_body += combine_predictions_ast
            return_ast = ast.parse("return __willump_final_predictions", "exec").body
            cascades_body += return_ast
            # Finalize AST.
            new_ast = copy.deepcopy(orig_ast)
            new_ast.body = cascades_body
            # No recursion allowed!
            new_ast.decorator_list = []
            return ast.copy_location(new_ast, orig_ast)

    func_source = inspect.getsource(func)
    func_ast = ast.parse(func_source)
    selected_feature_indices = cascades_dict["selected_feature_indices"]
    selected_feature_nodes = [model_node.input_nodes[i] for i in selected_feature_indices]
    selected_feature_names = [model_node.input_names[i] for i in selected_feature_indices]
    remaining_feature_nodes = [model_node.input_nodes[i] for i in range(len(model_node.input_nodes)) if
                               i not in selected_feature_indices]
    cascades_transformer = WillumpCascadesPredict()
    cascades_ast = cascades_transformer.visit(func_ast)
    cascades_ast = ast.fix_missing_locations(cascades_ast)
    # import astor
    # print(astor.to_source(cascades_ast))
    # Create namespaces the instrumented function can run in containing both its
    # original globals and the ones the instrumentation needs.
    local_namespace = {}
    augmented_globals = copy.copy(func.__globals__)
    augmented_globals["__willump_approximate_model"] = cascades_dict["approximate_model"]
    augmented_globals["__willump_full_model"] = cascades_dict["full_model"]
    augmented_globals["__willump_cascade_threshold"] = cascades_dict["cascade_threshold"]
    augmented_globals["__willump_get_unapproximated_indices"] = get_unapproximated_indices
    augmented_globals["__willump_select_unapproximated_rows"] = select_unapproximated_rows
    augmented_globals["__willump_combine_predictions"] = combine_predictions
    # Run the instrumented function.
    exec(compile(cascades_ast, filename="<ast>", mode="exec"), augmented_globals,
         local_namespace)
    return local_namespace[func.__name__]


def get_unapproximated_indices(approximated_preds, cascade_threshold):
    return np.logical_and(approximated_preds < cascade_threshold,
                          approximated_preds > 1 - cascade_threshold).nonzero()[0]


def select_unapproximated_rows(input_data, unapproximated_indices):
    if isinstance(input_data, scipy.sparse.csr.csr_matrix) or isinstance(input_data, np.ndarray):
        return input_data[unapproximated_indices]
    elif isinstance(input_data, pd.DataFrame):
        return input_data.iloc[unapproximated_indices]
    else:
        return input_data


def combine_predictions(approximate_predictions, full_predictions, cascade_threshold):
    final_predictions = np.zeros(approximate_predictions.shape, dtype=full_predictions.dtype)
    full_prediction_index = 0
    for i in range(len(final_predictions)):
        if approximate_predictions[i] > cascade_threshold:
            final_predictions[i] = 1
        elif approximate_predictions[i] < 1 - cascade_threshold:
            final_predictions[i] = 0
        else:
            final_predictions[i] = full_predictions[full_prediction_index]
            full_prediction_index += 1
    assert(full_prediction_index == len(full_predictions))
    return final_predictions
