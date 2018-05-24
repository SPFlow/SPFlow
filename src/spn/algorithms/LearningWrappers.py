'''
Created on March 30, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_cols_RDC, get_split_rows_RDC, get_split_cols_RDC_py, \
    get_split_rows_RDC_py

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf


def learn_mspn_with_missing(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3,
                            linear=False, ohe=False,
                            leaves=None, memory=None):
    if leaves is None:
        # leaves = create_histogram_leaf
        leaves = create_piecewise_leaf

    rand_gen = np.random.RandomState(17)

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe):
        split_cols = None
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, ohe=True, k=10, s=1 / 6,
                                               non_linearity=np.sin, n_jobs=1,
                                               rand_gen=rand_gen)
        if rows == "kmeans":
            split_rows = get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6,
                                               non_linearity=np.sin, n_jobs=1,
                                               rand_gen=rand_gen)
        elif rows == "rdc":
            split_rows = get_split_rows_RDC(ohe=ohe)

        if leaves is None:
            leaves = create_histogram_leaf

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)


def learn_mspn(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3, ohe=False,
               leaves=None, memory=None):
    if leaves is None:
        leaves = create_histogram_leaf

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols = None
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, ohe=ohe)
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(ohe=ohe)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)


from spn.algorithms.splitting.Random import get_split_cols_binary_random_partition, \
    get_split_rows_binary_random_partition, create_random_unconstrained_type_mixture_leaf


def learn_rand_spn(data, ds_context,
                   min_instances_slice=200,
                   row_a=2, row_b=5,
                   col_a=4, col_b=5,
                   col_threshold=0.6,
                   memory=None, rand_gen=None):
    def learn(data, ds_context, min_instances_slice, rand_gen):

        if rand_gen is None:
            rand_gen = np.random.RandomState(17)

        ds_context.rand_gen = rand_gen

        split_cols = get_split_cols_binary_random_partition(threshold=col_threshold,
                                                            beta_a=col_a, beta_b=col_b)
        splot_rows = get_split_rows_binary_random_partition(beta_a=row_a, beta_b=row_b)

        # leaves = create_random_parametric_leaf
        leaves = create_random_unconstrained_type_mixture_leaf

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, splot_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, min_instances_slice, rand_gen)
