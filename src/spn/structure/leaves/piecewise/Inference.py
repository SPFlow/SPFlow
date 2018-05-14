'''
Created on May 4, 2018

@author: Alejandro Molina
@author: Antonio Vergari
'''

import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood, add_node_mpe_likelihood
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear


def piecewise_log_likelihood(node, data, dtype=np.float64, context=None, node_log_likelihood=None):
    assert context is not None, "context is not none"

    probs = np.zeros((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = np.log(piecewise_complete_cases_likelihood(node, nd[~marg_ids], dtype=dtype, context=context))

    return probs


def piecewise_complete_cases_likelihood(node, obs, dtype=np.float64, context=None):
    probs = np.zeros((obs.shape[0], 1), dtype=dtype) + EPSILON
    domain = context.domains[node.scope[0]]
    lt = obs < (domain[0] - EPSILON)
    mt = obs > (domain[-1] + EPSILON)
    outside_domain = np.logical_or(lt, mt)
    assert outside_domain.sum() == 0, (obs[lt], obs[mt], domain)
    ivalues = np.interp(x=obs, xp=node.x_range, fp=node.y_range)
    ividx = ivalues > 0
    probs[ividx,0] = ivalues[ividx]
    return probs



def piecewise_mpe_likelihood(node, data, log_space=True, dtype=np.float64, context=None, node_mpe_likelihood=None):
    assert len(node.scope) == 1, node.scope

    log_probs = np.zeros((data.shape[0], 1), dtype=dtype)
    log_probs[:] = piecewise_log_likelihood(node, np.ones((1, data.shape[1])) * node.mode, dtype=dtype, context=context)

    #
    # collecting query rvs
    mpe_ids = np.isnan(data[:, node.scope[0]])

    log_probs[~mpe_ids] = piecewise_log_likelihood(node, data[~mpe_ids, :], dtype=dtype, context=context)

    if not log_space:
        return np.exp(log_probs)

    return log_probs




def add_piecewise_inference_support():
    add_node_likelihood(PiecewiseLinear, piecewise_log_likelihood)
    add_node_mpe_likelihood(PiecewiseLinear, piecewise_mpe_likelihood)