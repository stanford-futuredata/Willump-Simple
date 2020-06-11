import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
import pickle

base_directory = "tests/test_resources/music_resources/"


def auc_score(y_valid, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def music_train(y, X_list):
    X = pd.concat(X_list, axis=1, ignore_index=True)
    model = LGBMClassifier(
        n_jobs=1,
        learning_rate=0.1,
        num_leaves=(2 ** 8),
        max_depth=15,
        metric="auc")
    model = model.fit(X, y)
    return model


def music_predict(model, X_list):
    X = pd.concat(X_list, axis=1, ignore_index=True)
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    else:
        return model.predict(X)


def music_predict_proba(model, X_list):
    X = pd.concat(X_list, axis=1, ignore_index=True)
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
