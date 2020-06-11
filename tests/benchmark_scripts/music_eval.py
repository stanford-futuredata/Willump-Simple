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
    return music_predict(model, [features_uf, features_sf, uc_features, sc_features, ac_features,
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

    _, join_col_uf = load_als_dataframe(base_directory, size=UF_SIZE, user=True, artist=False)
    _, join_col_sf = load_als_dataframe(base_directory, size=SF_SIZE, user=False, artist=False)
    _, uc_join_col = scol_features_eval(base_directory, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    _, sc_join_col = scol_features_eval(base_directory, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    _, ac_join_col = scol_features_eval(base_directory, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    _, us_col = scol_features_eval(base_directory, 'msno', 'u_')
    _, ss_col = scol_features_eval(base_directory, 'song_id', 's_')
    _, as_col = scol_features_eval(base_directory, 'artist_name', 'a_')
    _, gs_col = scol_features_eval(base_directory, 'genre_max', 'gmax_')
    _, cs_col = scol_features_eval(base_directory, 'city', 'c_')
    _, ages_col = scol_features_eval(base_directory, 'bd', 'age_')
    _, ls_col = scol_features_eval(base_directory, 'language', 'lang_')
    _, gender_col = scol_features_eval(base_directory, 'gender', 'gen_')
    _, composer_col = scol_features_eval(base_directory, 'composer', 'comp_')
    _, lyrs_col = scol_features_eval(base_directory, 'lyricist', 'ly_')
    _, sns_col = scol_features_eval(base_directory, 'source_screen_name', 'sn_')
    _, stabs_col = scol_features_eval(base_directory, 'source_system_tab', 'sst_')
    _, stypes_col = scol_features_eval(base_directory, 'source_type', 'st_')
    _, regs_col = scol_features_eval(base_directory, 'registered_via', 'rv_')

    _, test_X, _, test_y = sklearn.model_selection.train_test_split(cluster_X, y, test_size=0.2, random_state=42)

    model = pickle.load(open(base_directory + "model.pk", "rb"))

    music_eval_pipeline(test_X.iloc[0:100], model)
    music_eval_pipeline(test_X.iloc[0:100], model)
    time_start = time.time()
    preds = music_eval_pipeline(test_X, model)
    time_elapsed = time.time() - time_start
    print("Elapsed Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, len(test_X), len(test_X) / time_elapsed))

    print("AUC Score: %f" % music_score(preds, test_y))
