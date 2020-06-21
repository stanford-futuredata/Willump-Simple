import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn import metrics
import pickle

base_directory = "tests/test_resources/music_resources/"


def auc_score(y_valid, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def music_train(y, X_list):
    X = pd.concat(X_list, axis=1)
    X = X[[f for f in FEATURES if f in X.columns]]
    model = LGBMClassifier(
        n_jobs=1,
        learning_rate=0.1,
        num_leaves=(2 ** 8),
        max_depth=15,
        metric="auc")
    model = model.fit(X, y)
    return model


def music_predict(model, X_list):
    X = pd.concat(X_list, axis=1)
    X = X[[f for f in FEATURES if f in X.columns]]
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    else:
        return model.predict(X)


def music_predict_proba(model, X_list):
    X = pd.concat(X_list, axis=1)
    X = X[[f for f in FEATURES if f in X.columns]]
    return model.predict_proba(X)[:, 1]


def music_score(true_y, pred_y):
    return auc_score(true_y, pred_y)


def get_features_from_redis(df, column, name, db):
    pipe = db.pipeline()
    for key in df[column].values:
        redis_key = name + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32
# CLUSTER SIZES
UC_SIZE = 25
SC_SIZE = 25
USC_SIZE = 25

LATENT_USER_FEATURES = ['uf_' + str(i) for i in range(UF_SIZE)]
LATENT_SONG_FEATURES = ['sf_' + str(i) for i in range(SF_SIZE)]
UC_FEATURES = [
    'uc_played',
    'uc_played_pos',
    'uc_played_rel',
    'uc_played_rel_global',
    'uc_played_pos_rel',
    'uc_played_ratio'
]
SC_FEATURES = [
    'sc_played',
    'sc_played_pos',
    'sc_played_rel',
    'sc_played_rel_global',
    'sc_played_ratio'
]

AC_FEATURES = [
    'ac_played',
    'ac_played_pos',
    'ac_played_rel',
    'ac_played_rel_global',
    'ac_played_ratio'
]
AGE_FEATURES = [
    'age_played',
    'age_played_pos',
    'age_played_rel',
    'age_played_rel_global',
    'age_played_pos_rel',
    'age_played_ratio'
]

REG_VIA_FEATURES = [
    'rv_played',
    'rv_played_pos',
    'rv_played_rel',
    'rv_played_rel_global',
    'rv_played_pos_rel',
    'rv_played_ratio'
]

LANGUAGE_FEATURES = [
    'lang_played',
    'lang_played_pos',
    'lang_played_rel',
    'lang_played_rel_global',
    'lang_played_pos_rel',
    'lang_played_ratio'
]

GENDER_FEATURES = [
    'gen_played',
    'gen_played_pos',
    'gen_played_rel',
    'gen_played_rel_global',
    'gen_played_pos_rel',
    'gen_played_ratio'
]
LYRICIST_FEATURES = [
    'ly_played',
    'ly_played_rel',
    'ly_played_rel_global',
]
COMPOSER_FEATURES = [
    'comp_played',
    'comp_played_rel',
    'comp_played_rel_global',
]

USER_FEATURES = [
    'u_played',
    'u_played_rel',
    'u_played_rel_global',
]

SONG_FEATURES = [
    's_played',
    's_played_rel',
    's_played_rel_global',
]

ARTIST_FEATURES = [
    'a_played',
    'a_played_rel',
    'a_played_rel_global',
]

SOURCE_NAME_FEATURES = [
    'sn_played',
    'sn_played_pos',
    'sn_played_rel',
    'sn_played_rel_global',
    'sn_played_pos_rel',
    'sn_played_ratio'
]

SOURCE_TAB_FEATURES = [
    'sst_played',
    'sst_played_pos',
    'sst_played_rel',
    'sst_played_rel_global',
    'sst_played_pos_rel',
    'sst_played_ratio'
]

SOURCE_TYPE_FEATURES = [
    'st_played',
    'st_played_pos',
    'st_played_rel',
    'st_played_pos_rel',
    'st_played_ratio'
]

CITY_FEATURES = [
    'c_played',
    'c_played_pos',
    'c_played_rel',
    'c_played_rel_global',
    'c_played_pos_rel',
    'c_played_ratio'
]

GENRE_FEATURES_MAX = [
    'gmax_played',
    'gmax_played_pos',
    'gmax_played_rel',
    'gmax_played_rel_global',
    'gmax_played_pos_rel',
    'gmax_played_ratio',
]

FEATURES = LATENT_SONG_FEATURES + LATENT_USER_FEATURES
FEATURES = FEATURES + SC_FEATURES
FEATURES = FEATURES + UC_FEATURES
FEATURES = FEATURES + AC_FEATURES

FEATURES = FEATURES + LANGUAGE_FEATURES  # pos included
FEATURES = FEATURES + AGE_FEATURES  # pos included
FEATURES = FEATURES + GENDER_FEATURES  # pos included
FEATURES = FEATURES + REG_VIA_FEATURES  # pos included
FEATURES = FEATURES + COMPOSER_FEATURES  # + COMPOSER_FEATURES_RATIO + COMPOSER_FEATURES_POS
FEATURES = FEATURES + LYRICIST_FEATURES  # + LYRICIST_FEATURES_RATIO #+ LYRICIST_FEATURES_POS

FEATURES = FEATURES + USER_FEATURES  # + USER_FEATURES_RATIO + USER_FEATURES_POS

FEATURES = FEATURES + SONG_FEATURES  # + SONG_FEATURES_RATIO #+ SONG_FEATURES_POS

FEATURES = FEATURES + ARTIST_FEATURES

FEATURES = FEATURES + GENRE_FEATURES_MAX

FEATURES = FEATURES + SOURCE_NAME_FEATURES  # pos included
FEATURES = FEATURES + SOURCE_TAB_FEATURES  # pos included
FEATURES = FEATURES + SOURCE_TYPE_FEATURES  # pos included
FEATURES = FEATURES + CITY_FEATURES  # pos included


def put_features_in_redis(df, name, db):
    for key in df.index:
        value = df.loc[key]
        ser_value = pickle.dumps(value)
        redis_key = name + "_" + str(int(key))
        db.set(redis_key, ser_value)


def load_music_dataset(redis=None):
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
    if redis is not None:
        put_features_in_redis(features_uf, "features_uf", redis)
    features_sf, join_col_sf = load_als_dataframe(base_directory, size=SF_SIZE, user=False, artist=False)
    features_sf = features_sf.set_index(join_col_sf).astype("float64")
    if redis is not None:
        put_features_in_redis(features_sf, "features_sf", redis)

    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features(base_directory, cluster_X, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    uc_features = uc_features.set_index(uc_join_col).astype("float64")
    if redis is not None:
        put_features_in_redis(uc_features, "uc_features", redis)
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features(base_directory, cluster_X, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    sc_features = sc_features.set_index(sc_join_col).astype("float64")
    if redis is not None:
        put_features_in_redis(sc_features, "sc_features", redis)
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features(base_directory, cluster_X, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    ac_features = ac_features.set_index(ac_join_col).astype("float64")
    if redis is not None:
        put_features_in_redis(ac_features, "ac_features", redis)
    # USER FEATURES
    us_features, us_col = scol_features(base_directory, X, 'msno', 'u_')
    us_features = us_features.set_index(us_col).astype("float64")
    if redis is not None:
        put_features_in_redis(us_features, "us_features", redis)
    # SONG FEATURES
    ss_features, ss_col = scol_features(base_directory, X, 'song_id', 's_')
    ss_features = ss_features.set_index(ss_col).astype("float64")
    if redis is not None:
        put_features_in_redis(ss_features, "ss_features", redis)
    # ARTIST FEATURES
    as_features, as_col = scol_features(base_directory, X, 'artist_name', 'a_')
    as_features = as_features.set_index(as_col).astype("float64")
    if redis is not None:
        put_features_in_redis(as_features, "as_features", redis)
    # GENRE FEATURES
    gs_features, gs_col = scol_features(base_directory, X, 'genre_max', 'gmax_')
    gs_features = gs_features.set_index(gs_col).astype("float64")
    if redis is not None:
        put_features_in_redis(gs_features, "gs_features", redis)
    # CITY FEATURES
    cs_features, cs_col = scol_features(base_directory, X, 'city', 'c_')
    cs_features = cs_features.set_index(cs_col).astype("float64")
    if redis is not None:
        put_features_in_redis(cs_features, "cs_features", redis)
    # AGE FEATURES
    ages_features, ages_col = scol_features(base_directory, X, 'bd', 'age_')
    ages_features = ages_features.set_index(ages_col).astype("float64")
    if redis is not None:
        put_features_in_redis(ages_features, "ages_features", redis)
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features(base_directory, X, 'language', 'lang_')
    ls_features = ls_features.set_index(ls_col).astype("float64")
    if redis is not None:
        put_features_in_redis(ls_features, "ls_features", redis)
    # GENDER FEATURES
    gender_features, gender_col = scol_features(base_directory, X, 'gender', 'gen_')
    gender_features = gender_features.set_index(gender_col).astype("float64")
    if redis is not None:
        put_features_in_redis(gender_features, "gender_features", redis)
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features(base_directory, X, 'composer', 'comp_')
    composer_features = composer_features.set_index(composer_col).astype("float64")
    if redis is not None:
        put_features_in_redis(composer_features, "composer_features", redis)
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features(base_directory, X, 'lyricist', 'ly_')
    lyrs_features = lyrs_features.set_index(lyrs_col).astype("float64")
    if redis is not None:
        put_features_in_redis(lyrs_features, "lyrs_features", redis)
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features(base_directory, X, 'source_screen_name', 'sn_')
    sns_features = sns_features.set_index(sns_col).astype("float64")
    if redis is not None:
        put_features_in_redis(sns_features, "sns_features", redis)
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features(base_directory, X, 'source_system_tab', 'sst_')
    stabs_features = stabs_features.set_index(stabs_col).astype("float64")
    if redis is not None:
        put_features_in_redis(stabs_features, "stabs_features", redis)
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features(base_directory, X, 'source_type', 'st_')
    stypes_features = stypes_features.set_index(stypes_col).astype("float64")
    if redis is not None:
        put_features_in_redis(stypes_features, "stypes_features", redis)
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features(base_directory, X, 'registered_via', 'rv_')
    regs_features = regs_features.set_index(regs_col).astype("float64")
    if redis is not None:
        put_features_in_redis(regs_features, "regs_features", redis)

    return sklearn.model_selection.train_test_split(cluster_X, y, test_size=0.2, random_state=42)


def load_combi_prep(folder='data_new/', split=None):
    name = 'combi_extra' + ('.' + str(split) if split is not None else '') + '.pkl'
    combi = pd.read_pickle(folder + name)
    return combi


def load_als_dataframe(folder, size, user, artist):
    if user:
        if artist:
            name = 'user2'
            key = 'uf2_'
        else:
            name = 'user'
            key = 'uf_'
    else:
        if artist:
            name = 'artist'
            key = 'af_'
        else:
            name = 'song'
            key = 'sf_'
    csv_name = folder + 'als' + '_' + name + '_features' + '.{}.csv'.format(size)
    features = pd.read_csv(csv_name)

    for i in range(size):
        features[key + str(i)] = features[key + str(i)].astype(np.float32)
    oncol = 'msno' if user else 'song_id' if not artist else 'artist_name'
    return features, oncol


def scol_features_eval(folder, col, prefix):
    csv_name = folder + prefix + 'cluster' + '_' + col + '_features' + '.csv'
    tmp = pd.read_csv(csv_name)
    return tmp, col


def add_cluster(folder, col, size, overlap=True, positive=True, content=False):
    name = 'cluster_' + col
    file_name = 'alsclusterEMB32_' + col
    if content:
        file_name = 'content_' + file_name
    if overlap:
        file_name += '_ol'
    if not positive:
        file_name += '_nopos'

    # cluster = pd.read_csv( folder + 'content_' + name +'.{}.csv'.format(size) )
    cluster = pd.read_csv(folder + file_name + '.{}.csv'.format(size))

    cluster[name + '_' + str(size)] = cluster.cluster_id
    del cluster['cluster_id']

    return cluster, col


def scol_features(folder, combi, col, prefix):
    tmp = pd.DataFrame()
    group = combi.groupby([col])
    group_pos = combi[combi.target == 1].groupby(col)
    tmp[prefix + 'played'] = group.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = group_pos.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = tmp[prefix + 'played_pos'].fillna(0)
    tmp[prefix + 'played_rel'] = (tmp[prefix + 'played'] / tmp[prefix + 'played'].max()).astype(np.float32)
    tmp[prefix + 'played_rel_global'] = (tmp[prefix + 'played'] / len(combi)).astype(np.float32)
    tmp[prefix + 'played_pos_rel'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played_pos'].max()).astype(np.float32)
    tmp[prefix + 'played_ratio'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played']).astype(np.float32)
    tmp[col] = tmp.index
    csv_name = folder + prefix + 'cluster' + '_' + col + '_features' + '.csv'
    tmp.to_csv(csv_name)
    return tmp, col


def compute_features(input_X, db):
    user_latent_features = get_features_from_redis(input_X, column="msno", name="features_uf", db=db)
    song_latent_features = get_features_from_redis(input_X, column="song_id", name="features_sf", db=db)
    user_cluster_features = get_features_from_redis(input_X, column="cluster_msno_25", name="uc_features", db=db)
    song_cluster_features = get_features_from_redis(input_X, column="cluster_song_id_25", name="sc_features", db=db)
    artist_cluster_features = get_features_from_redis(input_X, column="cluster_artist_name_25", name="ac_features",
                                                      db=db)
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
    features = [user_latent_features, song_latent_features, user_cluster_features, song_cluster_features,
                artist_cluster_features, user_features, song_features, artist_features, genre_features,
                city_features, ages_features, language_features, gender_features, composer_features,
                lyrs_features, sns_features, stabs_features, stypes_features,
                regs_features]
    feature_names = ["user_latent_features", "song_latent_features", "user_cluster_features", "song_cluster_features",
                     "artist_cluster_features", "user_features", "song_features", "artist_features", "genre_features",
                     "city_features", "ages_features", "language_features", "gender_features", "composer_features",
                     "lyrs_features", "sns_features", "stabs_features", "stypes_features",
                     "regs_features"]
    feature_costs = {name: 1.0 for name in feature_names}
    return features, feature_costs
