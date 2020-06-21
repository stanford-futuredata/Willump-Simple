import argparse

import redis
import sklearn
import time

from music_utils import *
from willump.evaluation.willump_executor import willump_execute

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
args = parser.parse_args()
if args.cascades:
    cascades_dict = pickle.load(open(base_directory + "training_cascades.pk", "rb"))
else:
    cascades_dict = None

db = redis.StrictRedis(host="localhost")

train_cascades_dict = {}


def put_features_in_redis(df, name):
    for key in df.index:
        value = df.loc[key]
        ser_value = pickle.dumps(value)
        redis_key = name + "_" + str(int(key))
        db.set(redis_key, ser_value)


@willump_execute(predict_function=music_predict,
                 predict_proba_function=music_predict_proba,
                 predict_cascades_dict=cascades_dict)
def music_eval_pipeline(input_X, model):
    user_latent_features = get_features_from_redis(input_X, column="msno", name="features_uf", db=db)
    song_latent_features = get_features_from_redis(input_X, column="song_id", name="features_sf", db=db)
    user_cluster_features = get_features_from_redis(input_X, column="cluster_msno_25", name="uc_features", db=db)
    song_cluster_features = get_features_from_redis(input_X, column="cluster_song_id_25", name="sc_features", db=db)
    artist_cluster_features = get_features_from_redis(input_X, column="cluster_artist_name_25", name="ac_features", db=db)
    user_features = get_features_from_redis(input_X, column="msno", name="us_features", db=db)
    song_features = get_features_from_redis(input_X, column="song_id", name="ss_features", db=db)
    artist_features = get_features_from_redis(input_X, column="artist_name", name="as_features", db=db)
    genre_features = get_features_from_redis(input_X, column="genre_max", name="gs_features", db=db)
    city_features = get_features_from_redis(input_X, column="city", name="cs_features", db=db)
    ages_features = get_features_from_redis(input_X, column="bd", name="ages_features", db=db)
    language_features = get_features_from_redis(input_X, column="language", name="ls_features", db=db)
    gender_features = get_features_from_redis(input_X, column="gender", name="gender_features", db=db)
    composer_features = get_features_from_redis(input_X, column="composer", name="composer_features", db=db)
    lyrs_features = get_features_from_redis(input_X, column="lyricist", name="lyrs_features", db=db)
    sns_features = get_features_from_redis(input_X, column="source_screen_name", name="sns_features", db=db)
    stabs_features = get_features_from_redis(input_X, column="source_system_tab", name="stabs_features", db=db)
    stypes_features = get_features_from_redis(input_X, column="source_type", name="stypes_features", db=db)
    regs_features = get_features_from_redis(input_X, column="registered_via", name="regs_features", db=db)
    return music_predict(model,
                         [user_latent_features, song_latent_features, user_cluster_features, song_cluster_features,
                          artist_cluster_features, user_features, song_features, artist_features, genre_features,
                          city_features, ages_features, language_features, gender_features, composer_features,
                          lyrs_features, sns_features, stabs_features, stypes_features,
                          regs_features])


if __name__ == '__main__':
    _, test_X, _, test_y = load_music_dataset(redis=None)

    model = pickle.load(open(base_directory + "model.pk", "rb"))

    music_eval_pipeline(test_X.iloc[0:100], model)
    music_eval_pipeline(test_X.iloc[0:100], model)
    time_start = time.time()
    preds = music_eval_pipeline(test_X, model)
    time_elapsed = time.time() - time_start
    print("Elapsed Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, len(test_X), len(test_X) / time_elapsed))

    print("AUC Score: %f" % music_score(preds, test_y))
