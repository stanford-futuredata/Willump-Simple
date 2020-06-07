import ast
import inspect
import unittest

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import willump.evaluation.construct_cascades
from willump.evaluation.willump_executor import instrument_function
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder

base_directory = "tests/test_resources/product_resources/"


def product_train(y, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    model = LogisticRegression(solver='liblinear')
    model = model.fit(x, y)
    return model


def product_predict(model, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        return model.predict(x)


def product_predict_proba(model, x_list):
    x = scipy.sparse.hstack([*x_list], format="csr")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    else:
        return model.predict_proba(x)[:, 1]


def product_score(true_y, pred_y):
    return 1 - np.sqrt(mean_squared_error(true_y, pred_y))


def transform_data(data, vectorizer):
    return vectorizer.transform(data["title"])


def product_train_pipeline(input_x, input_y, title_vect, color_vect, brand_vect):
    title_result = transform_data(input_x, title_vect)
    color_result = transform_data(input_x, color_vect)
    brand_result = transform_data(input_x, brand_vect)
    return product_train(input_y, [title_result, color_result, brand_result])


def product_predict_pipeline(input_x, model, title_vect, color_vect, brand_vect):
    title_result = transform_data(input_x, title_vect)
    color_result = transform_data(input_x, color_vect)
    brand_result = transform_data(input_x, brand_vect)
    return product_predict(model, [title_result, color_result, brand_result])


class CascadesTests(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv(base_directory + "lazada_data_train.csv", header=None,
                         names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                                'short_description', 'price', 'product_type'])
        y = np.loadtxt(base_directory + "conciseness_train.labels", dtype=int)
        self.train_df, self.test_df, self.train_y, self.test_y = train_test_split(df, y, test_size=0.2, random_state=42)
        self.title_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                                                lowercase=False, stop_words=None, binary=False, decode_error='replace')
        self.title_vectorizer.fit(self.train_df["title"].tolist())
        self.assertEqual(len(self.title_vectorizer.vocabulary_), 8801)

        with open(base_directory + "colors.txt", encoding="windows-1252") as color_file:
            colors = [x.strip() for x in color_file.readlines()]
            c = list(filter(lambda x: len(x.split()) > 1, colors))
            c = list(map(lambda x: x.replace(" ", ""), c))
            colors.extend(c)
            self.color_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace',
                                                    lowercase=False)
            self.color_vectorizer.fit(colors)
            self.assertEqual(len(self.color_vectorizer.vocabulary_), 1848)

        with open(base_directory + "brands_from_lazada_portal.txt", encoding="windows-1252") as brands_file:
            brands = [x.strip() for x in brands_file.readlines()]
            self.brand_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace',
                                                    lowercase=False)
            self.brand_vectorizer.fit(brands)
            self.assertEqual(len(self.brand_vectorizer.vocabulary_), 20274)

    def test_pipeline(self):
        model = product_train_pipeline(self.train_df, self.train_y, self.title_vectorizer, self.color_vectorizer,
                                       self.brand_vectorizer)
        preds = product_predict_pipeline(self.test_df, model, self.title_vectorizer, self.color_vectorizer,
                                         self.brand_vectorizer)
        self.assertAlmostEqual(product_score(preds, self.test_y), 0.570612, 6)

    def test_timer_graph_builder(self):
        timing_map, model_data = {}, {}
        instrumented_train = instrument_function(product_train_pipeline, timing_map, model_data)
        instrumented_predict = instrument_function(product_predict_pipeline, {}, {})
        model = instrumented_train(self.train_df, self.train_y, self.title_vectorizer, self.color_vectorizer,
                                   self.brand_vectorizer)
        preds = instrumented_predict(self.test_df, model, self.title_vectorizer, self.color_vectorizer,
                                     self.brand_vectorizer)
        self.assertAlmostEqual(product_score(preds, self.test_y), 0.570612, 6)
        self.assertTrue('title_result' in timing_map and 'color_result' in timing_map and 'brand_result' in timing_map)
        self.assertTrue((model_data['params'] == self.train_y).all())
        train_source = inspect.getsource(product_train_pipeline)
        train_ast = ast.parse(train_source)
        graph_builder = WillumpGraphBuilder(timing_map)
        graph_builder.visit(train_ast)
        model_node = graph_builder.get_model_node()
        self.assertEqual(model_node.function_name, "product_train")
        self.assertEqual(model_node.model_param, "input_y")
        self.assertEqual(model_node.input_names[0], "title_result")
        self.assertEqual(model_node.input_names[1], "color_result")
        self.assertEqual(model_node.input_names[2], "brand_result")
        color_node = model_node.input_nodes[1]
        self.assertEqual(color_node.function_name, "transform_data")
        self.assertEqual(color_node.output_name, "color_result")
        self.assertEqual(color_node.input_names[0], "input_x")
        self.assertEqual(color_node.input_names[1], "color_vect")

    def test_calculate_feature_importances(self):
        title_result = transform_data(self.train_df, self.title_vectorizer)
        color_result = transform_data(self.train_df, self.color_vectorizer)
        brand_result = transform_data(self.train_df, self.brand_vectorizer)
        X = [title_result, color_result, brand_result]
        train_X, valid_X, train_y, valid_y = \
            willump.evaluation.construct_cascades.train_test_split(X, self.train_y, test_size=0.25, random_state=42)
        train_set_full_model = product_train(train_y, train_X)
        feature_importances = \
            willump.evaluation.construct_cascades.calculate_feature_importances(
                train_set_full_model, valid_X, valid_y, product_predict, product_score,
                ["title_result", "color_result", "brand_result"])
        assert(feature_importances["title_result"] > feature_importances["brand_result"] > 0)
        assert(feature_importances["brand_result"] > feature_importances["color_result"])


if __name__ == '__main__':
    unittest.main()
