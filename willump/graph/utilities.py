import ast
from typing import List


def create_function_ast(function_name: str, output_name: str, input_names: List[str]) -> ast.AST:
    function_arguments: str = ""
    for input_name in input_names:
        function_arguments += "%s," % input_name
    function_statement = "%s = %s(%s)" % (output_name, function_name, function_arguments)
    return ast.parse(function_statement, "exec")


def create_model_ast(function_name: str, output_name: str, input_names: List[str], model_param: str) -> ast.AST:
    model_arguments: str = ""
    for input_name in input_names:
        model_arguments += "%s," % input_name
    model_statement = "%s = %s(%s, [%s])" % (output_name, function_name, model_param, model_arguments)
    return ast.parse(model_statement, "exec")