import ast
import copy
from typing import List


class WillumpRuntimeTimer(ast.NodeTransformer):

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_body: List[ast.stmt] = []
        for entry in node.body:
            if isinstance(entry, ast.Assign):
                # Start timing
                timing_start_code: str = \
                    """t0 = time.time()\n"""
                timing_start_ast: ast.Module = ast.parse(timing_start_code, "exec")
                timing_start_statement: List[ast.stmt] = timing_start_ast.body
                new_body = new_body + timing_start_statement
                # Original statement
                output_name: str = entry.targets[0].id
                new_body.append(entry)
                # End timing
                timing_end_code: str = \
                    """willump_timing_map["%s"] = time.time() - t0\n""" % output_name
                timing_end_ast: ast.Module = ast.parse(timing_end_code, "exec")
                timing_end_statement: List[ast.stmt] = timing_end_ast.body
                new_body = new_body + timing_end_statement
            elif isinstance(entry, ast.Return):
                # Record the model parameters
                model_param_name: str = entry.value.args[0].id
                model_inputs_list: str = ""
                for function_argument in entry.value.args[1].elts:
                    input_name = function_argument.id
                    model_inputs_list += "%s," % input_name
                model_inputs_list = "[" + model_inputs_list + "]"
                record_params_code: str = \
                    """willump_model_data["params"] = %s\nwillump_model_data["inputs"]=%s\n""" \
                    % (model_param_name, model_inputs_list)
                record_params_ast: ast.Module = ast.parse(record_params_code, "exec")
                record_params_statement: List[ast.stmt] = record_params_ast.body
                new_body = new_body + record_params_statement
                new_body.append(entry)
            else:
                print("Error:  Unrecognized Node detected: %s" % entry)
        new_node = copy.deepcopy(node)
        new_node.body = new_body
        # No recursion allowed!
        new_node.decorator_list = []
        return ast.copy_location(new_node, node)
