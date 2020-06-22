import argparse
import pickle
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from toxic_utils import *
from willump.evaluation.willump_executor import willump_execute

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
args = parser.parse_args()
if args.cascades:
    cascades_dict = pickle.load(open(base_directory + "training_cascades.pk", "rb"))
else:
    cascades_dict = None


@willump_execute(predict_function=toxic_predict,
                 predict_proba_function=toxic_predict_proba,
                 predict_cascades_params=cascades_dict)
def toxic_eval_pipeline(input_x, model, word_vect, char_vect):
    word_features = transform_data(input_x, word_vect)
    char_features = transform_data(input_x, char_vect)
    return toxic_predict(model, [word_features, char_features])


if __name__ == '__main__':
    df = pd.read_csv(base_directory + 'train.csv').fillna(' ')
    y = df["toxic"]
    X = df["comment_text"].values
    _, test_X, _, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    word_vectorizer, char_vectorizer = pickle.load(open(base_directory + "vectorizer.pk", "rb"))
    model = pickle.load(open(base_directory + "model.pk", "rb"))

    toxic_eval_pipeline(test_X, model, word_vectorizer, char_vectorizer)
    toxic_eval_pipeline(test_X, model, word_vectorizer, char_vectorizer)

    start_time = time.time()
    preds = toxic_eval_pipeline(test_X, model, word_vectorizer, char_vectorizer)
    time_elapsed = time.time() - start_time

    print("Elapsed Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, len(test_X), len(test_X) / time_elapsed))

    print("ROC-AUC Score: %f" % toxic_score(preds, test_y))



