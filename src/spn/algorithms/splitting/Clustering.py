'''
Created on March 25, 2018

@author: Alejandro Molina
'''
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc

_rpy_initialized = False


def init_rpy():
    global _rpy_initialized
    if _rpy_initialized:
        return
    _rpy_initialized = True

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os
    path = os.path.dirname(__file__)
    with open(path + "/mixedClustering.R", "r") as rfile:
        code = ''.join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
    #https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os
    ncpus = n_jobs
    if n_jobs < 1:
        ncpus = max(os.cpu_count() - 1, 1)

    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(data)
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(kmeans_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    def split_rows_DBScan(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_DBScan


def get_split_rows_Gower(n_clusters=2, pre_proc=None, seed=17):
    from rpy2 import robjects
    init_rpy()

    def split_rows_Gower(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, False)

        try:
            df = robjects.r["as.data.frame"](data)
            clusters = robjects.r["mixedclustering"](df, ds_context.distribution_family, n_clusters, seed)
            clusters = np.asarray(clusters)
        except Exception as e:
            np.savetxt("/tmp/errordata.txt", local_data)
            print(e)
            raise e

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Gower
