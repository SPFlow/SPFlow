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

    # dataIn = data[:, node.evidence_size:]
    dataIn = data[:, -node.evidence_size:]
    dataOut = data[:, node.scope[0]]

    probs = np.ones((data.shape[0], 1), dtype=dtype)

    # marginalize over something?
    marg_ids = np.isnan(dataOut)

    # print("evidence", node.evidence_size)
    # print("data", np.shape(data))
    # print("dataIn/Out shape", np.shape(dataIn), np.shape(dataOut))
    # print("dataIn[~marg_ids])", np.shape(dataIn[~marg_ids]))
    scipy_obj, params = get_scipy_obj_params(node, dataIn[~marg_ids])

    if isinstance(node, Conditional_Gaussian):
        # params is a vector instead of a scalar
        probs[~marg_ids, 0] = scipy_obj.pdf(dataOut[~marg_ids], **params)

    elif isinstance(node, Conditional_Poisson) or isinstance(node, Conditional_Bernoulli):
        probs[~marg_ids, 0] = scipy_obj.pmf(dataOut[~marg_ids], **params)

    else:
        raise Exception("Unknown parametric " + str(type(node)))

    # print(probs)
    return probs


def add_conditional_inference_support():
    add_node_likelihood(Conditional_Gaussian, conditional_likelihood)
    add_node_likelihood(Conditional_Poisson, conditional_likelihood)
    add_node_likelihood(Conditional_Bernoulli, conditional_likelihood)
