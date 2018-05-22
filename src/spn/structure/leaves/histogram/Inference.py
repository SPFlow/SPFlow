'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood, add_node_mpe_likelihood
from spn.structure.leaves.histogram.Histograms import Histogram
from numba import jit


# @jit("float64[:](float64[:], float64[:], float64[:,:])", nopython=True)
def histogram_ll(breaks, densities, data):
    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        if x < breaks[0] or x >= breaks[-1]:
            continue

        # TODO: binary search
        j = 0
        for b in breaks:
            if b > x:
                break
            j += 1

        probs[i] = densities[j - 1]

    probs[probs < EPSILON] = EPSILON

    return probs


def histogram_log_likelihood(node, data, dtype=np.float64, context=None, node_log_likelihood=None):
    probs = np.zeros((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = np.log(histogram_ll(np.array(node.breaks), np.array(node.densities), nd[~marg_ids]))

    return probs


def histogram_mpe_log_likelihood(node, data, log_space=True, dtype=np.float64, context=None, node_mpe_likelihood=None):
    assert len(node.scope) == 1, node.scope

    log_probs = np.zeros((data.shape[0], 1), dtype=dtype)
    log_probs[:] = histogram_log_likelihood(node, np.ones((1, data.shape[1])) * node.mode, dtype=dtype)

    #
    # collecting query rvs
    mpe_ids = np.isnan(data[:, node.scope[0]])

    log_probs[~mpe_ids] = histogram_log_likelihood(node, data[~mpe_ids, :], dtype=dtype)

    if not log_space:
        return np.exp(log_probs)

    return log_probs


def add_histogram_inference_support():
    add_node_likelihood(Histogram, histogram_log_likelihood)

    add_node_mpe_likelihood(Histogram, histogram_mpe_log_likelihood)
