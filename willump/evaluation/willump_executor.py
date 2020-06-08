import ast
import copy
import importlib
import inspect
from typing import Callable, MutableMapping, Mapping

from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump.evaluation.willump_runtime_timer import WillumpRuntimeTimer
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.evaluation.cascades_construct import construct_cascades
from willump.evaluation.cascades_predict import predict_cascades

timing_map_set: MutableMapping[str, MutableMapping[str, float]] = {}
model_data_set: MutableMapping[str, MutableMapping[str, object]] = {}
willump_final_func_set: MutableMapping[str, Callable] = {}


def instrument_function(func: Callable, timing_map: MutableMapping[str, float],
                        model_data: MutableMapping[str, object]) -> Callable:
    python_source = inspect.getsource(func)
    python_ast: ast.AST = ast.parse(python_source)
    function_name: str = python_ast.body[0].name
    type_discover: WillumpRuntimeTimer = WillumpRuntimeTimer()
    # Create an instrumented AST that will time all operators in the function.
    new_ast: ast.AST = type_discover.visit(python_ast)
    new_ast = ast.fix_missing_locations(new_ast)
    # import astor
    # print(astor.to_source(new_ast))
    # Create namespaces the instrumented function can run in containing both its
    # original globals and the ones the instrumentation needs.
    local_namespace = {}
    augmented_globals = copy.copy(func.__globals__)
    augmented_globals["willump_timing_map"] = timing_map
    augmented_globals["willump_model_data"] = model_data
    augmented_globals["time"] = importlib.import_module("time")
    # Run the instrumented function.
    exec(compile(new_ast, filename="<ast>", mode="exec"), augmented_globals,
         local_namespace)
    return local_namespace[function_name]


def willump_execute(train_function: Callable = None, predict_function: Callable = None,
                    predict_proba_function: Callable = None, score_function: Callable = None,
                    train_cascades_dict: MutableMapping = None,
                    predict_cascades_dict: Mapping = None) -> Callable:
    def willump_execute_inner(func: Callable) -> Callable:
        func_id: str = "willump_func_id%s" % func.__name__

        def function_wrapper(*args):
            if func_id not in timing_map_set:
                timing_map_set[func_id] = {}
                model_data_set[func_id] = {}
                instrumented_func: Callable = \
                    instrument_function(func, timing_map_set[func_id], model_data_set[func_id])
                return instrumented_func(*args)
            elif func_id not in willump_final_func_set:
                timing_map = timing_map_set[func_id]
                model_data = model_data_set[func_id]
                function_source = inspect.getsource(func)
                function_ast = ast.parse(function_source)
                graph_builder = WillumpGraphBuilder(timing_map)
                graph_builder.visit(function_ast)
                model_node: WillumpGraphNode = graph_builder.get_model_node()
                if train_cascades_dict is not None:
                    construct_cascades(model_data,
                                       model_node,
                                       train_function, predict_function,
                                       predict_proba_function, score_function,
                                       train_cascades_dict)
                    return train_cascades_dict["full_model"]
                elif predict_cascades_dict is not None:
                    cascades_func = predict_cascades(func,
                                                     model_node,
                                                     predict_function,
                                                     predict_proba_function,
                                                     predict_cascades_dict)
                    willump_final_func_set[func_id] = cascades_func
                    return cascades_func(*args)
                else:
                    willump_final_func_set[func_id] = func
                    return func(*args)
            else:
                return willump_final_func_set[func_id](*args)

        return function_wrapper

    return willump_execute_inner
