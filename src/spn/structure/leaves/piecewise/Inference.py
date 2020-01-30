"""
Created on May 4, 2018

@author: Alejandro Molina
@author: Antonio Vergari
"""

import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood, leaf_marginalized_likelihood
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
import logging

logger = logging.getLogger(__name__)

LOG_ZERO = -300


def piecewise_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    probs, marg_ids, observations = leaf_marginalized_likelihood(node, data, dtype, log_space=True)
    probs[~marg_ids] = piecewise_complete_cases_log_likelihood(node, observations, dtype=dtype)
    return probs


def _compute_probability_for_range(node, interval):
    lower = interval[0]
    higher = interval[1]

    x_range = np.array(node.x_range)
    y_range = np.array(node.y_range)

    lower_prob = np.interp(lower, xp=x_range, fp=y_range)
    higher_prob = np.interp(higher, xp=x_range, fp=y_range)

    indicies = np.where((lower <= x_range) & (x_range <= higher))

    x_tmp = [lower] + list(x_range[indicies]) + [higher]
    y_tmp = [lower_prob] + list(y_range[indicies]) + [higher_prob]

    return np.trapz(y_tmp, x_tmp)


def piecewise_complete_cases_log_likelihood(node, obs, dtype=np.float64):
    probs = np.ones((obs.shape[0]), dtype=dtype)  # + EPSILON
    ivalues = np.interp(x=obs, xp=node.x_range, fp=node.y_range)
    probs[:] = ivalues
    # ividx = ivalues > 0
    # probs[ividx, 0] = ivalues[ividx]
    assert np.all(probs >= 0.0)
    return np.log(probs)


def add_piecewise_inference_support():
    add_node_likelihood(PiecewiseLinear, log_lambda_func=piecewise_log_likelihood)
