'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.misc import logsumexp

from spn.structure.Base import Product, Sum, Leaf
from spn.structure.leaves.Histograms import Histogram

EPSILON = 0.000000000000001


def histogram_likelihood(node, data, log_space=True, dtype=np.float64):
    probs = np.zeros((data.shape[0]), dtype=dtype)

    for i, x in enumerate(data):
        if x < node.breaks[0] or x > node.breaks[-1]:
            continue

        # TODO: binary search
        j = 0
        for b in node.breaks:
            if b > x:
                break
            j += 1

        probs[i] = node.densities[j - 1]

    probs[probs < EPSILON] = EPSILON

    if log_space:
        return np.log(probs)

    return probs


def log_likelihood(node, data, log_space=True, dtype=np.float64):
    if isinstance(node, Histogram):
        return histogram_likelihood(node, data[:, node.scope], log_space, dtype)

    if isinstance(node, Product):
        llchildren = np.zeros((data.shape[0]), dtype=dtype)

        # TODO: parallelize here
        for c in node.children:
            llchildren[:] += log_likelihood(c, data, log_space, dtype)

        if log_space:
            return llchildren

        return np.exp(llchildren)

    if isinstance(node, Sum):
        llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

        # TODO: parallelize here
        for i, c in enumerate(node.children):
            llchildren[:, i] = log_likelihood(c, data, log_space, dtype)

        b = np.array(node.weights, dtype=dtype).reshape(1, -1)

        ll = logsumexp(llchildren, b=b, axis=1)

        if log_space:
            return ll

        return np.exp(ll)

    raise Exception('Node type unknown: ' + str(type(node)))


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, True, dtype) - log_likelihood(node_marginal, data, True, dtype)
    if log_space:
        return result

    return np.exp(result)
