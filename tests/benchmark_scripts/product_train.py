import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from product_utils import *
from willump.evaluation.willump_executor import willump_execute
import pickle

train_cascades_dict = {}


@willump_execute(train_function=product_train,
                 predict_function=product_predict,
                 predict_proba_function=product_predict_proba,
                 score_function=product_score,
                 train_cascades_dict=train_cascades_dict)
def product_train_pipeline(input_x, input_y, title_vect, color_vect, brand_vect):
    title_result = transform_data(input_x, title_vect)
    color_result = transform_data(input_x, color_vect)
    brand_result = transform_data(input_x, brand_vect)
    return product_train(input_y, [title_result, color_result, brand_result])


if __name__ == '__main__':
    df = pd.read_csv(base_directory + "lazada_data_train.csv", header=None,
                     names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                            'short_description', 'price', 'product_type'])
    y = np.loadtxt(base_directory + "conciseness_train.labels", dtype=int)
    train_df, _, train_y, _ = train_test_split(df, y, test_size=0.2, random_state=42)
    title_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                                       lowercase=False, stop_words=None, binary=False, decode_error='replace')
    title_vectorizer.fit(train_df["title"].tolist())

    with open(base_directory + "colors.txt", encoding="windows-1252") as color_file:
        colors = [x.strip() for x in color_file.readlines()]
        c = list(filter(lambda x: len(x.split()) > 1, colors))
        c = list(map(lambda x: x.replace(" ", ""), c))
        colors.extend(c)
        color_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace',
                                           lowercase=False)
        color_vectorizer.fit(colors)

    with open(base_directory + "brands_from_lazada_portal.txt", encoding="windows-1252") as brands_file:
        brands = [x.strip() for x in brands_file.readlines()]
        brand_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace',
                                           lowercase=False)
        brand_vectorizer.fit(brands)
        product_train_pipeline(train_df, train_y, title_vectorizer, color_vectorizer, brand_vectorizer)
        model = product_train_pipeline(train_df, train_y, title_vectorizer, color_vectorizer, brand_vectorizer)
        pickle.dump((title_vectorizer, color_vectorizer, brand_vectorizer),
                    open(base_directory + "lazada_vectorizers.pk", "wb"))
        pickle.dump(model, open(base_directory + "lazada_model.pk", "wb"))
        pickle.dump(train_cascades_dict,
                    open(base_directory + "lazada_training_cascades.pk", "wb"))
