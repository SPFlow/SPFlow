'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.misc import logsumexp

from spn.structure.Base import Product, Sum, get_nodes_by_type
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


def likelihood(node, data, log_space=True, dtype=np.float64, node_likelihood=None, lls_matrix=None):
    if node_likelihood is not None:
        t_node = type(node)
        if t_node in node_likelihood:
            ll = node_likelihood[t_node](node, data, log_space, dtype, node_likelihood)
            if lls_matrix is not None:
                lls_matrix[:, node.id] = ll
            return ll

    if isinstance(node, Histogram):
        ll = histogram_likelihood(node, data[:, node.scope], log_space, dtype)
        if lls_matrix is not None:
            lls_matrix[:, node.id] = ll
        return ll

    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    # TODO: parallelize here
    for i, c in enumerate(node.children):
        llchildren[:, i] = likelihood(c, data, log_space, dtype, node_likelihood, lls_matrix)

    if isinstance(node, Product):
        ll = np.sum(llchildren, axis=1)

        if not log_space:
            ll = np.exp(llchildren)

    elif isinstance(node, Sum):
        b = np.array(node.weights, dtype=dtype).reshape(1, -1)

        ll = logsumexp(llchildren, b=b, axis=1)

        if not log_space:
            ll = np.exp(llchildren)
    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    if lls_matrix is not None:
        lls_matrix[:, node.id] = ll

    return ll




def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = likelihood(node_joint, data, True, dtype) - likelihood(node_marginal, data, True, dtype)
    if log_space:
        return result

    return np.exp(result)


def full_likelihood(node, data, log_space=True, dtype=np.float64, node_likelihood=None, lls=None):
    if lls is None:
        nr_nodes = len(get_nodes_by_type(node))
        lls = np.zeros((data.shape[0], nr_nodes)) / 0

    if node_likelihood is not None:
        t_node = type(node)
        if t_node in node_likelihood:
            return node_likelihood[t_node](node, data, log_space, dtype, node_likelihood)

    if isinstance(node, Histogram):
        return histogram_likelihood(node, data[:, node.scope], log_space, dtype)

    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    for i, c in enumerate(node.children):
        llchildren[:, i] = full_likelihood(c, data, log_space, dtype, node_likelihood, lls)

    if isinstance(node, Product):
        if log_space:
            return llchildren

        return np.exp(llchildren)

    if isinstance(node, Sum):
        b = np.array(node.weights, dtype=dtype).reshape(1, -1)

        ll = logsumexp(llchildren, b=b, axis=1)

        if log_space:
            return ll

        return np.exp(ll)

    raise Exception('Node type unknown: ' + str(type(node)))