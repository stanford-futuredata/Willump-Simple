import ast
from typing import List
import willump.graph.utilities


class WillumpGraphNode:

    def __init__(self, function_name: str, output_name: str, input_names: List[str],
                 input_nodes: List['WillumpGraphNode'], cost: float, model_param: str = None):
        self.function_name = function_name
        self.output_name = output_name
        self.input_names = input_names
        self.input_nodes = input_nodes
        self.cost = cost
        assert(len(input_nodes) == len(input_names))
        self.model_param = model_param

    def get_ast(self) -> ast.AST:
        if self.model_param is None:
            return willump.graph.utilities.create_function_ast(function_name=self.function_name,
                                                               output_name=self.output_name,
                                                               input_names=self.input_names)
        else:
            return willump.graph.utilities.create_model_ast(function_name=self.function_name,
                                                            output_name=self.output_name,
                                                            input_names=self.input_names,
                                                            model_param=self.model_param)
