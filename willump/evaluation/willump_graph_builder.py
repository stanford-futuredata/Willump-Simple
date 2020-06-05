import ast
from typing import MutableMapping, Mapping

from willump.graph.willump_graph_node import WillumpGraphNode


class WillumpGraphBuilder(ast.NodeVisitor):
    _variable_dict: MutableMapping[str, WillumpGraphNode] = {}
    _model_node: WillumpGraphNode

    def __init__(self, timing_map: Mapping[str, float]):
        self._timing_map = timing_map

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for entry in node.body:
            if isinstance(entry, ast.Assign):
                self.analyze_assign(entry)
            elif isinstance(entry, ast.Return):
                self.analyze_return(entry)
            else:
                print("Error:  Unrecognized Node detected: %s" % entry)

    def analyze_assign(self, node: ast.Assign) -> None:
        assert (len(node.targets) == 1)  # Assume assignment to only one variable.
        output_name: str = node.targets[0].id
        function_name: str = node.value.func.id
        input_names, input_nodes = [], []
        for function_argument in node.value.args:
            input_name = function_argument.id
            input_names.append(input_name)
            if input_name in self._variable_dict:
                input_nodes.append(self._variable_dict[input_name])
            else:
                input_nodes.append(None)
        function_node: WillumpGraphNode = WillumpGraphNode(function_name=function_name,
                                                           output_name=output_name,
                                                           input_names=input_names,
                                                           input_nodes=input_nodes,
                                                           cost=self._timing_map[output_name])
        self._variable_dict[output_name] = function_node

    def analyze_return(self, node: ast.Return) -> None:
        output_name: str = "__willump_return_result"
        function_name: str = node.value.func.id
        assert (len(node.value.args) == 2)  # First is the model parameter, second is a list of model arguments.
        model_param: str = node.value.args[0].id
        input_names, input_nodes = [], []
        for function_argument in node.value.args[1].elts:
            input_name = function_argument.id
            input_names.append(input_name)
            if input_name in self._variable_dict:
                input_nodes.append(self._variable_dict[input_name])
            else:
                input_nodes.append(None)
        model_node: WillumpGraphNode = WillumpGraphNode(function_name=function_name,
                                                        output_name=output_name,
                                                        input_names=input_names,
                                                        input_nodes=input_nodes,
                                                        model_param=model_param,
                                                        cost=0)
        self._model_node = model_node

    def get_model_node(self):
        return self._model_node
