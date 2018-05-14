'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.Inference import add_node_likelihood, add_node_mpe_likelihood
from spn.structure.leaves.parametric.Parametric import *
from spn.structure.leaves.parametric.utils import get_scipy_obj_params


POS_EPS = 1e-7

LOG_ZERO = -300


def parametric_log_likelihood(node, data, dtype=np.float64, context=None, node_log_likelihood=None):
    assert len(node.scope) == 1, node.scope

    log_probs = np.zeros((data.shape[0], 1), dtype=dtype)

    if data.shape[1] > 1:
        data = data[:, node.scope]

    assert data.shape[1] == 1, data.shape

    #
    # marginalize over something?
    marg_ids = np.isnan(data)

    if isinstance(node, Gaussian) or isinstance(node, LogNormal) or \
            isinstance(node, Exponential):
        scipy_obj, params = get_scipy_obj_params(node)
        log_probs[~marg_ids] = scipy_obj.logpdf(data[~marg_ids], **params)
        # if np.any(np.isposinf(log_probs[~marg_ids])):
        #     inf_ids = np.isposinf(log_probs)
        #     print(node, node.scope, log_probs[inf_ids],
        #           node.params, data[~marg_ids], data[inf_ids], params)
        #     0 / 0
    elif isinstance(node, Gamma):
        scipy_obj, params = get_scipy_obj_params(node)
        data_m = data[~marg_ids]
        data_m[data_m == 0] += POS_EPS
        log_probs[~marg_ids] = scipy_obj.logpdf(data_m, **params)
        # if np.any(np.isposinf(log_probs[~marg_ids])):
        #     inf_ids = np.isposinf(log_probs)
        #     print(node, node.scope, log_probs[inf_ids],
        #           node.params, data[~marg_ids], data[inf_ids], params)
        #     0 / 0

    elif isinstance(node, Poisson) or isinstance(node, Bernoulli) or isinstance(node, Geometric):
        scipy_obj, params = get_scipy_obj_params(node)
        log_probs[~marg_ids] = scipy_obj.logpmf(data[~marg_ids], **params)
        # if np.any(np.isposinf(log_probs[~marg_ids])):
        #     inf_log = np.isposinf(log_probs)
        #     print(log_probs[inf_log], data[inf_log])
        #     print(data[~marg_ids], (~marg_ids).sum(),  log_probs[~marg_ids])
        #     0 / 0
    elif isinstance(node, NegativeBinomial):
        raise ValueError('Mismatch with scipy')
    elif isinstance(node, Hypergeometric):
        raise ValueError('Mismatch with wiki')
    elif isinstance(node, Categorical):
        #
        # forcing casting
        cat_data = data.astype(np.int64)
        assert np.all(np.equal(np.mod(cat_data[~marg_ids], 1), 0))
        # assert np.all(np.logical_and(cat_data[~marg_ids] >= 0, cat_data[~marg_ids] < node.k))
        out_domain_ids = cat_data >= node.k
        log_probs[~marg_ids & out_domain_ids] = LOG_ZERO
        log_probs[~marg_ids & ~out_domain_ids] = np.array(
            np.log(node.p))[cat_data[~marg_ids & ~out_domain_ids]]
    elif isinstance(node, Uniform):
        log_probs[~marg_ids] = np.log(node.density)
    else:
        raise Exception("Unknown parametric " + str(type(node)))

    return log_probs


def parametric_mpe_log_likelihood(node, data, log_space=True, dtype=np.float64, context=None, node_mpe_likelihood=None):
    assert len(node.scope) == 1, node.scope

    log_probs = np.zeros((data.shape[0], 1), dtype=dtype)
    log_probs[:] = parametric_log_likelihood(node, np.array([[node.mode]]), dtype=dtype)

    if data.shape[1] > 1:
        data = data[:, node.scope]

    assert data.shape[1] == 1, data.shape

    #
    # collecting query rvs
    mpe_ids = np.isnan(data)

    log_probs[~mpe_ids] = parametric_log_likelihood(
        node, data[~mpe_ids].reshape(-1, 1), dtype=dtype)[:, 0]

    if not log_space:
        return np.exp(log_probs)

    return log_probs


def add_parametric_inference_support():
    add_node_likelihood(Gaussian, parametric_log_likelihood)
    add_node_likelihood(Gamma, parametric_log_likelihood)
    add_node_likelihood(LogNormal, parametric_log_likelihood)
    add_node_likelihood(Poisson, parametric_log_likelihood)
    add_node_likelihood(Bernoulli, parametric_log_likelihood)
    add_node_likelihood(Categorical, parametric_log_likelihood)
    add_node_likelihood(NegativeBinomial, parametric_log_likelihood)
    add_node_likelihood(Hypergeometric, parametric_log_likelihood)
    add_node_likelihood(Geometric, parametric_log_likelihood)
    add_node_likelihood(Exponential, parametric_log_likelihood)
    add_node_likelihood(Uniform, parametric_log_likelihood)

    add_node_mpe_likelihood(Gaussian, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Gamma, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(LogNormal, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Poisson, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Bernoulli, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Categorical, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(NegativeBinomial, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Hypergeometric, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Geometric, parametric_mpe_log_likelihood)
    add_node_mpe_likelihood(Exponential, parametric_mpe_log_likelihood)
