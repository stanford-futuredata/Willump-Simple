import ast
import inspect
import unittest

from willump.evaluation.willump_executor import willump_execute, instrument_function
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump.graph.willump_graph_node import WillumpGraphNode


def foo(a, one):
    return a + one


def bar(b):
    return b + 2


def model(m, numbers):
    return m + sum(numbers)


def foobar(a, b, m):
    c = foo(a, one=1)
    d = bar(b)
    return model(m, [c, d])


@willump_execute()
def foobar_decorated(a, b, m):
    c = foo(a, one=1)
    d = bar(b)
    return model(m, [c, d])


class GraphTests(unittest.TestCase):
    def test_graph_builder(self):
        foobar_source = inspect.getsource(foobar)
        foobar_ast = ast.parse(foobar_source)
        graph_builder = WillumpGraphBuilder({'c': 1, 'd': 1})
        graph_builder.visit(foobar_ast)
        model_node: WillumpGraphNode = graph_builder.get_model_node()
        self.assertEqual(model_node.function_name, "model")
        self.assertEqual(model_node.model_param, "m")
        self.assertEqual(model_node.input_names[0], "c")
        self.assertEqual(model_node.input_names[1], "d")
        foo_node: WillumpGraphNode = model_node.input_nodes[0]
        bar_node: WillumpGraphNode = model_node.input_nodes[1]
        self.assertEqual(foo_node.function_name, "foo")
        self.assertEqual(foo_node.output_name, "c")
        self.assertEqual(foo_node.input_names[0], "a")
        self.assertEqual(bar_node.function_name, "bar")
        self.assertEqual(bar_node.output_name, "d")
        self.assertEqual(bar_node.input_names[0], "b")

    def test_timer_instrumentation(self):
        timing_map = {}
        model_data = {}
        instrumented_foobar = instrument_function(foobar, timing_map, model_data)
        self.assertEqual(instrumented_foobar(1, 2, 3), 9)
        self.assertTrue('c' in timing_map and 'd' in timing_map)
        self.assertEqual(model_data['params'], 3)
        self.assertEqual(model_data['inputs'], [2, 4])

    def test_decorator(self):
        self.assertEqual(foobar_decorated(1, 2, 3), 9)
        self.assertEqual(foobar_decorated(1, 2, 3), 9)
        self.assertEqual(foobar_decorated(1, 2, 3), 9)
        self.assertEqual(foobar_decorated(1, 2, 3), 9)


if __name__ == '__main__':
    unittest.main()
