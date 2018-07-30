'''
Created on March 30, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.Validity import is_valid
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.algorithms.splitting.RDC import get_split_rows_RDC, get_split_cols_RDC_py, \
    get_split_rows_RDC_py
from spn.algorithms.splitting.Random import get_split_cols_binary_random_partition, \
    get_split_rows_binary_random_partition
from spn.algorithms.splitting.RCoT import getCIGroups

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.conditional.Conditional import create_conditional_leaf


def learn_classifier(data, ds_context, spn_learn_wrapper, label_idx, cpus=-1, rand_gen=None):
    spn = Sum()
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        branch = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, cpus=cpus, rand_gen=rand_gen)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)
    assign_ids(spn)

    valid, err = is_valid(spn)
    assert valid, "invalid spn: " + err

    return spn


def learn_mspn_with_missing(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3,
                            linear=False, ohe=False, leaves=None, memory=None, rand_gen=None, cpus=-1):
    if leaves is None:
        # leaves = create_histogram_leaf
        leaves = create_piecewise_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()

        if leaves is None:
            leaves = create_histogram_leaf

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)


def learn_mspn(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3, ohe=False,
               leaves=None, memory=None, rand_gen=None, cpus=-1):
    if leaves is None:
        leaves = create_histogram_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)


def learn_parametric(data, ds_context, cols="rdc", rows="kmeans", min_instances_slice=200, threshold=0.3, ohe=False,
                     leaves=None, memory=None, rand_gen=None, cpus=-1):
    if leaves is None:
        leaves = create_parametric_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)



def learn_conditional(data, ds_context, scope=None, cols="ci", rows="rand_hp", min_instances_slice=200, threshold=0.3, ohe=False,
                     leaves=None, memory=None):
    """
    :param data: np array
    :param ds_context: Context object
    :param scope: list of indices of output variables
    :param cols: column splitting method
    :param rows: row splitting method
    :param min_instances_slice: minimal instance slice
    :param threshold: threshold scalar
    :param ohe: ohe
    :param leaves: boolean
    :param memory: boolean
    :return: method to learn structure
    """
    if leaves is None:
        leaves = create_conditional_leaf

    def learn(data, ds_context, scope, cols, rows, min_instances_slice, threshold, ohe):
        split_cols = None
        if cols == "ci":
            split_cols = getCIGroups(data, scope, threshold)
        else:
            raise ValueError, 'invalid independence test'
        if rows == "rand_hp":
            split_rows = get_split_rows_binary_random_partition(ohe=ohe)
        else:
            #todo add other clustering?
            raise ValueError, 'invalid clustering method'

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop, scope)

    if memory:
        learn = memory.cache(learn)

    return learn(data, ds_context, scope, cols, rows, min_instances_slice, threshold, ohe)