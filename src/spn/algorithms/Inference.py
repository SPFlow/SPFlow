"""
Created on March 21, 2018

@author: Alejandro Molina
"""
import numpy as np
from scipy.special import logsumexp

from spn.structure.Base import Product, Sum, Leaf, eval_spn_bottom_up, eval_spn_top_down

EPSILON = 0.000000000000001


def leaf_marginalized_likelihood(node, data=None, dtype=np.float64):
    assert len(node.scope) == 1, node.scope
    probs = np.ones((data.shape[0], 1), dtype=dtype)
    assert data.shape[1] >= 1
    data = data[:, node.scope]
    marg_ids = np.isnan(data)
    observations = data[~marg_ids]
    assert len(observations.shape) == 1, observations.shape
    return probs, marg_ids, observations


def prod_log_likelihood(node, children, data=None, dtype=np.float64):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    return np.sum(llchildren, axis=1).reshape(-1, 1)


def prod_likelihood(node, children, data=None, dtype=np.float64):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    return np.prod(llchildren, axis=1).reshape(-1, 1)


def sum_log_likelihood(node, children, data=None, dtype=np.float64):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)


def sum_likelihood(node, children, data=None, dtype=np.float64):
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return np.dot(llchildren, b).reshape(-1, 1)


_node_log_likelihood = {Sum: sum_log_likelihood, Product: prod_log_likelihood}
_node_likelihood = {Sum: sum_likelihood, Product: prod_likelihood}


def log_node_likelihood(node, *args, **kwargs):
    probs = _node_likelihood[type(node)](node, *args, **kwargs)
    with np.errstate(divide="ignore"):
        return np.log(probs)


def add_node_likelihood(node_type, lambda_func, log_lambda_func=None):
    _node_likelihood[node_type] = lambda_func
    if log_lambda_func is None:
        log_lambda_func = log_node_likelihood
    _node_log_likelihood[node_type] = log_lambda_func


_node_mpe_likelihood = {}


def add_node_mpe_likelihood(node_type, lambda_func):
    _node_mpe_likelihood[node_type] = lambda_func


def likelihood(node, data, dtype=np.float64, node_likelihood=_node_likelihood, lls_matrix=None, debug=False):
    all_results = {}

    if debug:
        assert len(data.shape) == 2, "data must be 2D, found: {}".format(data.shape)
        original_node_likelihood = node_likelihood

        def exec_funct(node, *args, **kwargs):
            assert node is not None, "node is nan "
            funct = original_node_likelihood[type(node)]
            ll = funct(node, *args, **kwargs)
            assert ll.shape == (data.shape[0], 1), "node %s result has to match dimensions (N,1)" % node.id
            assert not np.any(np.isnan(ll)), "ll is nan %s " % node.id
            return ll

        node_likelihood = {k: exec_funct for k in node_likelihood.keys()}

    result = eval_spn_bottom_up(node, node_likelihood, all_results=all_results, debug=debug, dtype=dtype, data=data)

    if lls_matrix is not None:
        for n, ll in all_results.items():
            lls_matrix[:, n.id] = ll[:, 0]

    return result


def log_likelihood(
    node, data, dtype=np.float64, node_log_likelihood=_node_log_likelihood, lls_matrix=None, debug=False
):
    return likelihood(node, data, dtype=dtype, node_likelihood=node_log_likelihood, lls_matrix=lls_matrix, debug=debug)


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, dtype) - log_likelihood(node_marginal, data, dtype)
    if log_space:
        return result

    return np.exp(result)
