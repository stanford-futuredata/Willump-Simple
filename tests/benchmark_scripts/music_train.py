import pickle

import redis
import sklearn

from music_utils import *
from willump.evaluation.willump_executor import willump_execute

db = redis.StrictRedis(host="localhost")

train_cascades_dict = {}


def put_features_in_redis(df, name):
    for key in df.index:
        value = df.loc[key]
        ser_value = pickle.dumps(value)
        redis_key = name + "_" + str(int(key))
        db.set(redis_key, ser_value)


@willump_execute(train_function=music_train,
                 predict_function=music_predict,
                 predict_proba_function=music_predict_proba,
                 score_function=music_score,
                 train_cascades_dict=train_cascades_dict)
def music_train_pipeline(input_X, input_y):
    features_uf = get_features_from_redis(input_X, column=join_col_uf, name="features_uf", db=db)
    features_sf = get_features_from_redis(input_X, column=join_col_sf, name="features_sf", db=db)
    uc_features = get_features_from_redis(input_X, column=uc_join_col, name="uc_features", db=db)
    sc_features = get_features_from_redis(input_X, column=sc_join_col, name="sc_features", db=db)
    ac_features = get_features_from_redis(input_X, column=ac_join_col, name="ac_features", db=db)
    us_features = get_features_from_redis(input_X, column=us_col, name="us_features", db=db)
    ss_features = get_features_from_redis(input_X, column=ss_col, name="ss_features", db=db)
    as_features = get_features_from_redis(input_X, column=as_col, name="as_features", db=db)
    gs_features = get_features_from_redis(input_X, column=gs_col, name="gs_features", db=db)
    cs_features = get_features_from_redis(input_X, column=cs_col, name="cs_features", db=db)
    ages_features = get_features_from_redis(input_X, column=ages_col, name="ages_features", db=db)
    ls_features = get_features_from_redis(input_X, column=ls_col, name="ls_features", db=db)
    gender_features = get_features_from_redis(input_X, column=gender_col, name="gender_features", db=db)
    composer_features = get_features_from_redis(input_X, column=composer_col, name="composer_features", db=db)
    lyrs_features = get_features_from_redis(input_X, column=lyrs_col, name="lyrs_features", db=db)
    sns_features = get_features_from_redis(input_X, column=sns_col, name="sns_features", db=db)
    stabs_features = get_features_from_redis(input_X, column=stabs_col, name="stabs_features", db=db)
    stypes_features = get_features_from_redis(input_X, column=stypes_col, name="stypes_features", db=db)
    regs_features = get_features_from_redis(input_X, column=regs_col, name="regs_features", db=db)
    return music_train(input_y, [features_uf, features_sf, uc_features, sc_features, ac_features,
                                 us_features, ss_features, as_features, gs_features, cs_features,
                                 ages_features, ls_features, gender_features, composer_features,
                                 lyrs_features, sns_features, stabs_features, stypes_features,
                                 regs_features])


if __name__ == '__main__':
    X = load_combi_prep(folder=base_directory, split=None)
    X = X.dropna(subset=["target"])

    y = X["target"].values

    cluster_one, join_col_cluster_one = add_cluster(base_directory, col='msno', size=UC_SIZE, overlap=True,
                                                    positive=True, content=False)
    cluster_two, join_col_cluster_two = add_cluster(base_directory, col='song_id', size=SC_SIZE, overlap=True,
                                                    positive=True, content=False)
    cluster_three, join_col_cluster_three = add_cluster(base_directory, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    cluster_X = X.merge(cluster_one, how='left', on=join_col_cluster_one)
    cluster_X = cluster_X.merge(cluster_two, how='left', on=join_col_cluster_two)
    cluster_X = cluster_X.merge(cluster_three, how='left', on=join_col_cluster_three)

    features_uf, join_col_uf = load_als_dataframe(base_directory, size=UF_SIZE, user=True, artist=False)
    features_uf = features_uf.set_index(join_col_uf).astype("float64")
    put_features_in_redis(features_uf, "features_uf")
    features_sf, join_col_sf = load_als_dataframe(base_directory, size=SF_SIZE, user=False, artist=False)
    features_sf = features_sf.set_index(join_col_sf).astype("float64")
    put_features_in_redis(features_sf, "features_sf")

    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features(base_directory, cluster_X, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    uc_features = uc_features.set_index(uc_join_col).astype("float64")
    put_features_in_redis(uc_features, "uc_features")
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features(base_directory, cluster_X, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    sc_features = sc_features.set_index(sc_join_col).astype("float64")
    put_features_in_redis(sc_features, "sc_features")
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features(base_directory, cluster_X, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    ac_features = ac_features.set_index(ac_join_col).astype("float64")
    put_features_in_redis(ac_features, "ac_features")
    # USER FEATURES
    us_features, us_col = scol_features(base_directory, X, 'msno', 'u_')
    us_features = us_features.set_index(us_col).astype("float64")
    put_features_in_redis(us_features, "us_features")
    # SONG FEATURES
    ss_features, ss_col = scol_features(base_directory, X, 'song_id', 's_')
    ss_features = ss_features.set_index(ss_col).astype("float64")
    put_features_in_redis(ss_features, "ss_features")
    # ARTIST FEATURES
    as_features, as_col = scol_features(base_directory, X, 'artist_name', 'a_')
    as_features = as_features.set_index(as_col).astype("float64")
    put_features_in_redis(as_features, "as_features")
    # GENRE FEATURES
    gs_features, gs_col = scol_features(base_directory, X, 'genre_max', 'gmax_')
    gs_features = gs_features.set_index(gs_col).astype("float64")
    put_features_in_redis(gs_features, "gs_features")
    # CITY FEATURES
    cs_features, cs_col = scol_features(base_directory, X, 'city', 'c_')
    cs_features = cs_features.set_index(cs_col).astype("float64")
    put_features_in_redis(cs_features, "cs_features")
    # AGE FEATURES
    ages_features, ages_col = scol_features(base_directory, X, 'bd', 'age_')
    ages_features = ages_features.set_index(ages_col).astype("float64")
    put_features_in_redis(ages_features, "ages_features")
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features(base_directory, X, 'language', 'lang_')
    ls_features = ls_features.set_index(ls_col).astype("float64")
    put_features_in_redis(ls_features, "ls_features")
    # GENDER FEATURES
    gender_features, gender_col = scol_features(base_directory, X, 'gender', 'gen_')
    gender_features = gender_features.set_index(gender_col).astype("float64")
    put_features_in_redis(gender_features, "gender_features")
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features(base_directory, X, 'composer', 'comp_')
    composer_features = composer_features.set_index(composer_col).astype("float64")
    put_features_in_redis(composer_features, "composer_features")
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features(base_directory, X, 'lyricist', 'ly_')
    lyrs_features = lyrs_features.set_index(lyrs_col).astype("float64")
    put_features_in_redis(lyrs_features, "lyrs_features")
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features(base_directory, X, 'source_screen_name', 'sn_')
    sns_features = sns_features.set_index(sns_col).astype("float64")
    put_features_in_redis(sns_features, "sns_features")
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features(base_directory, X, 'source_system_tab', 'sst_')
    stabs_features = stabs_features.set_index(stabs_col).astype("float64")
    put_features_in_redis(stabs_features, "stabs_features")
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features(base_directory, X, 'source_type', 'st_')
    stypes_features = stypes_features.set_index(stypes_col).astype("float64")
    put_features_in_redis(stypes_features, "stypes_features")
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features(base_directory, X, 'registered_via', 'rv_')
    regs_features = regs_features.set_index(regs_col).astype("float64")
    put_features_in_redis(regs_features, "regs_features")

    train_X, _, train_y, _ = sklearn.model_selection.train_test_split(cluster_X, y, test_size=0.2, random_state=42)

    music_train_pipeline(train_X, train_y)

    model = music_train_pipeline(train_X, train_y)
    pickle.dump(model, open(base_directory + "model.pk", "wb"))
    pickle.dump(train_cascades_dict, open(base_directory + "training_cascades.pk", "wb"))
