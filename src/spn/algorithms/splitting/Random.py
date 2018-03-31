'''
Created on March 20, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc


def make_planes(N, dim):
    result = np.zeros((N, dim))
    for i in range(N):
        result[i, :] = np.random.uniform(-1, 1, dim)

    return result / np.sqrt(np.sum(result * result, axis=1))[:, None]


def above(planes, data):
    nD = data.shape[0]
    nP = planes.shape[0]
    centered = data - np.mean(data, axis=0)
    result = np.zeros((nD, nP))
    for i in range(nD):
        for j in range(nP):
            result[i, j] = np.sum(planes[j, :] * centered[i, :]) > 0
    return result

def get_split_cols_random_partition(ohe=False):
    def split_cols_random_partitions(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, None, ohe)
        clusters = above(make_planes(1, local_data.shape[1]), data)[:, 0]

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_random_partitions

