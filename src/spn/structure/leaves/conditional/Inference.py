'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.Inference import add_node_likelihood, add_node_mpe_likelihood

from spn.structure.leaves.conditional.Conditional import *
from spn.structure.leaves.conditional.utils import get_scipy_obj_params

POS_EPS = 1e-7

LOG_ZERO = -300


def conditional_likelihood(node, data, scope, dtype=np.float64):
    """
    :param node: the query
    :param data: data including conditional columns
    :param scope: scope indicates the output columns in data
    :param dtype: data type
    :return: conditional likelihood
    """
    assert len(node.scope) == 1, node.scope

    idx = scope[0]
    dataOut = data[:, idx]
    dataIn = data[:, ~idx]

    probs = np.ones((dataOut.shape[0], 1), dtype=dtype)

    if dataOut.shape[1] > 1:
        dataOut = dataOut[:, node.scope]

    assert dataOut.shape[1] == 1, dataOut.shape

    # marginalize over something?
    marg_ids = np.isnan(dataOut)

    if isinstance(node, Conditional_Gaussian):
        scipy_obj, params = get_scipy_obj_params(node, dataIn)   # params is a vector instead of a scalar
        probs[~marg_ids] = [scipy_obj.pdf(obs, **param) for obs, param in zip(dataOut[~marg_ids], params)]  #todo double check


    elif isinstance(node, Conditional_Poisson) or isinstance(node, Conditional_Bernoulli):
        scipy_obj, params = get_scipy_obj_params(node, dataIn)
        probs[~marg_ids] = [scipy_obj.pmf(obs[~marg_ids], **param) for obs, param in zip(dataOut[~marg_ids], params)] # todo double check

    else:
        raise Exception("Unknown parametric " + str(type(node)))

    return probs


# todo rewrite?
def conditional_mpe_log_likelihood(node, data, scope=None, log_space=True, dtype=np.float64, context=None, node_mpe_likelihood=None):
    """
    :param node:
    :param data:
    :param scope:
    :param log_space:
    :param dtype:
    :param context:
    :param node_mpe_likelihood:
    :return:
    """
    assert len(node.scope) == 1, node.scope
    dataOut, dataIn = data

    log_probs = np.zeros((dataOut.shape[0], 1), dtype=dtype)
    log_probs[:] = parametric_log_likelihood(node, np.array([[node.mode]]), dtype=dtype)  # todo where is parametric_log_likelihood?

    if dataOut.shape[1] > 1:
        dataOut = dataOut[:, node.scope]

    assert dataOut.shape[1] == 1, dataOut.shape

    #
    # collecting query rvs
    mpe_ids = np.isnan(dataOut)

    log_probs[~mpe_ids] = parametric_log_likelihood(
        node, dataOut[~mpe_ids].reshape(-1, 1), dtype=dtype)[:, 0]   # todo where is parametric_log_likelihood?

    if not log_space:
        return np.exp(log_probs)

    return log_probs


def add_parametric_inference_support():
    add_node_likelihood(Conditional_Gaussian, conditional_likelihood)
    add_node_likelihood(Conditional_Poisson, conditional_likelihood)
    add_node_likelihood(Conditional_Bernoulli, conditional_likelihood)

    add_node_mpe_likelihood(Conditional_Gaussian, conditional_mpe_log_likelihood)
    add_node_mpe_likelihood(Conditional_Poisson, conditional_mpe_log_likelihood)
    add_node_mpe_likelihood(Conditional_Bernoulli, conditional_mpe_log_likelihood)
