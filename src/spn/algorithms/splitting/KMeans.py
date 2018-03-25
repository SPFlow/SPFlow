'''
Created on March 25, 2018

@author: Alejandro Molina
'''
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer

from spn.algorithms.splitting.Base import split_data_by_clusters
import numpy as np

def preptfidf(data):
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(data)

def preplog(data):
    return np.log(data + 1)

def prepsqrt(data):
    return np.sqrt(data)

def split_rows_KMeans(local_data, ds_context, scope, n_clusters=2, pre_proc=None, seed=17):

    data = np.copy(local_data)

    f = None
    if pre_proc == "tf-idf":
        f = preptfidf
    elif ds_context == "log+1":
        f = preplog
    elif ds_context == "sqrt":
        f = prepsqrt

    if f:
        data[:, ds_context.distribution_family == "poisson"] = f(data[:, ds_context.distribution_family == "poisson"])

    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(data)

    return split_data_by_clusters(local_data, clusters, scope, rows=True)

