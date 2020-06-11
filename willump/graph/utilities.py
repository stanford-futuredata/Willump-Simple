import ast
from typing import List


def create_function_ast(function_name: str, output_name: str, input_names: List[str], keywords: List = None) -> ast.AST:
    function_arguments: str = ""
    for input_name in input_names:
        function_arguments += "%s," % input_name
    function_statement = "%s = %s(%s)" % (output_name, function_name, function_arguments)
    function_ast = ast.parse(function_statement, "exec").body
    if keywords is not None:
        function_ast[0].value.keywords = keywords
    return function_ast


def create_model_ast(function_name: str, output_name: str, input_names: List[str], model_param: str) -> ast.AST:
    model_arguments: str = ""
    for input_name in input_names:
        model_arguments += "%s," % input_name
    model_statement = "%s = %s(%s, [%s])" % (output_name, function_name, model_param, model_arguments)
    return ast.parse(model_statement, "exec").body
