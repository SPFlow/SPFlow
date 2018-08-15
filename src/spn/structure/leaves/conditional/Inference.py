'''
Created on April 15, 2018

@author: Alejandro Molina
'''

import numpy as np

from spn.algorithms.Inference import add_node_likelihood, add_node_mpe_likelihood

from spn.structure.leaves.conditional.Conditional import *
from spn.structure.leaves.conditional.utils import get_scipy_obj_params


def conditional_likelihood(node, data, dtype=np.float64):
    """
    :param node: the query
    :param data: data including conditional columns
    :param dtype: data type
    :return: conditional likelihood
    """
    assert len(node.scope) == 1, node.scope

    dataIn = data[:, node.evidence_size:]
    dataOut = data[:, node.scope[0]]

    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # marginalize over something?
    marg_ids = np.isnan(dataOut)

    scipy_obj, params = get_scipy_obj_params(node, dataIn[~marg_ids])

    if isinstance(node, Conditional_Gaussian):
        # params is a vector instead of a scalar
        probs[~marg_ids,0] = scipy_obj.pdf(dataOut[~marg_ids], **params)

    elif isinstance(node, Conditional_Poisson) or isinstance(node, Conditional_Bernoulli):
        probs[~marg_ids,0] = scipy_obj.pmf(dataOut[~marg_ids], **params)

    else:
        raise Exception("Unknown parametric " + str(type(node)))

    return probs


# todo rewrite?
def conditional_mpe_log_likelihood(node, data, scope=None, log_space=True, dtype=np.float64, context=None,
                                   node_mpe_likelihood=None):
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
    log_probs[:] = parametric_log_likelihood(node, np.array([[node.mode]]),
                                             dtype=dtype)  # todo where is parametric_log_likelihood?

    if dataOut.shape[1] > 1:
        dataOut = dataOut[:, node.scope]

    assert dataOut.shape[1] == 1, dataOut.shape

    #
    # collecting query rvs
    mpe_ids = np.isnan(dataOut)

    log_probs[~mpe_ids] = parametric_log_likelihood(
        node, dataOut[~mpe_ids].reshape(-1, 1), dtype=dtype)[:, 0]  # todo where is parametric_log_likelihood?

    if not log_space:
        return np.exp(log_probs)

    return log_probs


def add_conditional_inference_support():
    add_node_likelihood(Conditional_Gaussian, conditional_likelihood)
    add_node_likelihood(Conditional_Poisson, conditional_likelihood)
    add_node_likelihood(Conditional_Bernoulli, conditional_likelihood)

    add_node_mpe_likelihood(Conditional_Gaussian, conditional_mpe_log_likelihood)
    add_node_mpe_likelihood(Conditional_Poisson, conditional_mpe_log_likelihood)
    add_node_mpe_likelihood(Conditional_Bernoulli, conditional_mpe_log_likelihood)
