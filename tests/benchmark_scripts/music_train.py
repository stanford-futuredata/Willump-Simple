import pickle

import redis

from music_utils import get_features_from_redis, music_train, music_predict, music_confidence, music_score, \
    load_music_dataset, base_directory
from willump.evaluation.willump_executor import willump_execute

db = redis.StrictRedis(host="localhost")

train_cascades_dict = {}


@willump_execute(train_function=music_train,
                 predict_function=music_predict,
                 confidence_function=music_confidence,
                 score_function=music_score,
                 train_cascades_params=train_cascades_dict)
def music_train_pipeline(input_X, input_y):
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
    return music_train(input_y,
                       [user_latent_features, song_latent_features, user_cluster_features, song_cluster_features,
                        artist_cluster_features, user_features, song_features, artist_features, genre_features,
                        city_features, ages_features, language_features, gender_features, composer_features,
                        lyrs_features, sns_features, stabs_features, stypes_features,
                        regs_features])


if __name__ == '__main__':
    train_X, _, train_y, _ = load_music_dataset(redis=db)

    music_train_pipeline(train_X, train_y)

    model = music_train_pipeline(train_X, train_y)
    pickle.dump(model, open(base_directory + "model.pk", "wb"))
    pickle.dump(train_cascades_dict, open(base_directory + "training_cascades.pk", "wb"))
