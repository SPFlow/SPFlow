"""
Created on March 20, 2018

@author: Alejandro Molina
"""

import numpy as np

from spn.algorithms.splitting.Base import split_data_by_clusters, preproc


def make_planes(N, dim, rand_gen):
    result = np.zeros((N, dim))
    for i in range(N):
        result[i, :] = rand_gen.uniform(-1, 1, dim)

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


def get_split_rows_random_partition(rand_gen, ohe=False):
    def split_rows_random_partitions(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, None, ohe)
        clusters = above(make_planes(1, local_data.shape[1], rand_gen), data)[:, 0]

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_random_partitions


def get_split_cols_random_partition(rand_gen, ohe=False):
    def split_cols_random_partitions(local_data, ds_context, scope):
        # same as above, but transpose the data
        data = preproc(local_data.T, ds_context, None, ohe)
        clusters = above(make_planes(1, data.shape[1], rand_gen), data)[:, 0]

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_random_partitions


def get_split_cols_binary_random_partition(threshold, rand_gen, beta_a=4, beta_b=5):
    """
    Randomly partitions the columns into two clusters with percentage threshold
    (otherwise does not split)
    The percentage of splitting is drawn from a Beta distribution with parameters (beta_a, beta_b)
    """

    def split_cols_binary_random_partitions(local_data, ds_context, scope):
        # data = preproc(local_data, ds_context, None, ohe)

        #
        # with a certain percentage it may fail, such that row partitioning may happen
        clusters = None
        p = rand_gen.rand()
        # print('P', p)
        if p > threshold:
            #
            # draw percentage of split from  a Beta
            alloc_perc = rand_gen.beta(a=beta_a, b=beta_b)
            clusters = rand_gen.choice(2, size=local_data.shape[1], p=[alloc_perc, 1 - alloc_perc])
            # print(clusters, clusters.sum(), clusters.shape, alloc_perc)
        else:
            clusters = np.zeros(local_data.shape[1])

        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_binary_random_partitions


def get_split_rows_binary_random_partition(rand_gen, beta_a=2, beta_b=5):
    """
    The percentage of splitting is drawn from a Beta distribution with parameters (beta_a, beta_b)

    """

    def split_rows_binary_random_partition(local_data, ds_context, scope):
        # data = preproc(local_data, ds_context, pre_proc, ohe)

        # draw percentage of split from  a Beta
        alloc_perc = rand_gen.beta(a=beta_a, b=beta_b)
        clusters = rand_gen.choice(2, size=local_data.shape[0], p=[alloc_perc, 1 - alloc_perc])

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_binary_random_partition
