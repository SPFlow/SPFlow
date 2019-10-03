"""
Created on March 30, 2018

@author: Alejandro Molina
"""

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.CnetStructureLearning import get_next_operation_cnet, learn_structure_cnet
from spn.algorithms.Validity import is_valid

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.algorithms.splitting.Conditioning import (
    get_split_rows_naive_mle_conditioning,
    get_split_rows_random_conditioning,
)
import logging

logger = logging.getLogger(__name__)


def learn_classifier(data, ds_context, spn_learn_wrapper, label_idx, **kwargs):
    spn = Sum()
    for label, count in zip(*np.unique(data[:, label_idx], return_counts=True)):
        branch = spn_learn_wrapper(data[data[:, label_idx] == label, :], ds_context, **kwargs)
        spn.children.append(branch)
        spn.weights.append(count / data.shape[0])

    spn.scope.extend(branch.scope)
    assign_ids(spn)

    valid, err = is_valid(spn)
    assert valid, "invalid spn: " + err

    return spn


def get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs):
    from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py

    if isinstance(cols, str):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows


def learn_mspn_with_missing(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    threshold=0.3,
    linear=False,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        # leaves = create_histogram_leaf
        leaves = create_piecewise_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def l_mspn_missing(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        l_mspn_missing = memory.cache(l_mspn_missing)

    return l_mspn_missing(data, ds_context, cols, rows, min_instances_slice, threshold, linear, ohe)


def learn_mspn(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    threshold=0.3,
    initial_scope=None,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        leaves = create_histogram_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop, initial_scope=initial_scope)

    if memory:
        l_mspn = memory.cache(l_mspn)

    return l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)


def learn_parametric(
    data,
    ds_context,
    cols="rdc",
    rows="kmeans",
    min_instances_slice=200,
    min_features_slice=1,
    multivariate_leaf=False,
    threshold=0.3,
    initial_scope=None,
    ohe=False,
    leaves=None,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    if leaves is None:
        leaves = create_parametric_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols, split_rows = get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice, min_features_slice, multivariate_leaf)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop, initial_scope=initial_scope)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)


def learn_cnet(
    data,
    ds_context,
    cond="naive_mle",
    min_instances_slice=200,
    min_features_slice=1,
    memory=None,
    rand_gen=None,
    cpus=-1,
):
    leaves = create_cltree_leaf

    if cond == "naive_mle":
        conditioning = get_split_rows_naive_mle_conditioning()
    elif cond == "random":
        conditioning = get_split_rows_random_conditioning()

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, conditioning, min_instances_slice):
        nextop = get_next_operation_cnet(min_instances_slice, min_features_slice)
        return learn_structure_cnet(data, ds_context, conditioning, leaves, nextop)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, conditioning, min_instances_slice)
