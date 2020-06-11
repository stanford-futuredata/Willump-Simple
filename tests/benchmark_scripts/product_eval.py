import argparse
import pickle
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from product_utils import *
from willump.evaluation.willump_executor import willump_execute

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
args = parser.parse_args()
if args.cascades:
    cascades_dict = pickle.load(open(base_directory + "lazada_training_cascades.pk", "rb"))
else:
    cascades_dict = None


@willump_execute(predict_function=product_predict,
                 predict_proba_function=product_predict_proba,
                 predict_cascades_dict=cascades_dict)
def product_eval_pipeline(input_x, model, title_vect, color_vect, brand_vect):
    title_result = transform_data(input_x, title_vect)
    color_result = transform_data(input_x, color_vect)
    brand_result = transform_data(input_x, brand_vect)
    return product_predict(model, [title_result, color_result, brand_result])


if __name__ == '__main__':
    df = pd.read_csv(base_directory + "lazada_data_train.csv", header=None,
                     names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                            'short_description', 'price', 'product_type'])
    y = np.loadtxt(base_directory + "conciseness_train.labels", dtype=int)
    _, test_df, _, test_y = train_test_split(df, y, test_size=0.2, random_state=42)
    title_vectorizer, color_vectorizer, brand_vectorizer = pickle.load(
        open(base_directory + "lazada_vectorizers.pk", "rb"))
    model = pickle.load(open(base_directory + "lazada_model.pk", "rb"))

    product_eval_pipeline(test_df, model, title_vectorizer, color_vectorizer, brand_vectorizer)
    product_eval_pipeline(test_df, model, title_vectorizer, color_vectorizer, brand_vectorizer)

    start_time = time.time()
    preds = product_eval_pipeline(test_df, model, title_vectorizer, color_vectorizer, brand_vectorizer)
    time_elapsed = time.time() - start_time

    print("Elapsed Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, len(test_df), len(test_df) / time_elapsed))

    print("1 - RMSE Score: %f" % product_score(preds, test_y))
