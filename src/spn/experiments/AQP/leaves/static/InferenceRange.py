"""
Created on June 21, 2018

@author: Moritz
"""

import numpy as np

from spn.algorithms.Inference import add_node_likelihood
from spn.experiments.AQP.leaves.static.StaticNumeric import StaticNumeric


def static_likelihood_range(node, ranges, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope

    probs = np.ones((ranges.shape[0], 1), dtype=dtype)
    ranges = ranges[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            probs[i] = 0

        # Compute the sum of the probability of all possible values
        probs[i] = sum([_compute_probability_for_range(node, interval) for interval in rang.get_ranges()])

    return probs


def _compute_probability_for_range(node, interval):

    if len(interval) == 1:
        if node.val == interval[0]:
            return 1
        else:
            return 0
    else:
        lower = interval[0]
        higher = interval[1]

        if lower <= node.val and node.val <= higher:
            return 1
        else:
            return 0


def add_static_inference_range_support():
    add_node_likelihood(StaticNumeric, static_likelihood_range)
