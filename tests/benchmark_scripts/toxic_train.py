import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from toxic_utils import *
from willump.evaluation.willump_executor import willump_execute

train_cascades_dict = {}


@willump_execute(train_function=toxic_train,
                 predict_function=toxic_predict,
                 confidence_function=toxic_confidence,
                 score_function=toxic_score,
                 train_cascades_params=train_cascades_dict)
def toxic_train_pipeline(input_x, input_y, word_vect, char_vect):
    word_features = transform_data(input_x, word_vect)
    char_features = transform_data(input_x, char_vect)
    return toxic_train(input_y, [word_features, char_features])


if __name__ == '__main__':
    df = pd.read_csv(base_directory + 'train.csv').fillna(' ')
    y = df["toxic"]
    X = df["comment_text"].values
    train_X, _, train_y, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    word_vectorizer = TfidfVectorizer(
        lowercase=False,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 1),
        encoding="ascii",
        decode_error="strict",
        max_features=10000)
    word_vectorizer.fit(train_X)
    char_vectorizer = TfidfVectorizer(
        lowercase=False,
        analyzer='char',
        ngram_range=(2, 6),
        encoding="ascii",
        decode_error="strict",
        max_features=50000)
    char_vectorizer.fit(train_X)
    toxic_train_pipeline(train_X, train_y, word_vectorizer, char_vectorizer)
    model = toxic_train_pipeline(train_X, train_y, word_vectorizer, char_vectorizer)

    pickle.dump(model, open(base_directory + "model.pk", "wb"))
    pickle.dump(train_cascades_dict, open(base_directory + "training_cascades.pk", "wb"))
    pickle.dump((word_vectorizer, char_vectorizer), open(base_directory + "vectorizer.pk", "wb"))


