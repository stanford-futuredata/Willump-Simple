import ast
import inspect
from typing import Callable, MutableMapping

from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump.graph.willump_graph_node import WillumpGraphNode

willump_timing_map_set: MutableMapping[str, MutableMapping[str, float]] = {}
willump_final_func_set: MutableMapping[str, Callable] = {}


def willump_execute() -> Callable:
    def willump_execute_inner(func: Callable) -> Callable:
        func_id: str = "willump_func_id%s" % func.__name__

        def function_wrapper(*args):
            if func_id not in willump_timing_map_set:
                willump_timing_map_set[func_id] = {}
                return func(*args)
            elif func_id not in willump_final_func_set:
                function_source = inspect.getsource(func)
                function_ast = ast.parse(function_source)
                graph_builder = WillumpGraphBuilder()
                graph_builder.visit(function_ast)
                model_node: WillumpGraphNode = graph_builder.get_model_node()
                willump_final_func_set[func_id] = func
                return func(*args)
            else:
                return willump_final_func_set[func_id](*args)
        return function_wrapper
    return willump_execute_inner

