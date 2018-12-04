"""
Created on June 21, 2018

@author: Moritz
"""

import numpy as np


def identity_likelihood_range(node, ranges, dtype=np.float64, **kwargs):
    assert len(node.scope) == 1, node.scope

    probs = np.zeros((ranges.shape[0], 1), dtype=dtype)
    ranges = ranges[:, node.scope[0]]

    for i, rang in enumerate(ranges):

        # Skip if no range is specified aka use a log-probability of 0 for that instance
        if rang is None:
            probs[i] = 1
            continue

        # Skip if no values for the range are provided
        if rang.is_impossible():
            continue

        for interval in rang.get_ranges():

            if len(interval) == 1:
                lower = np.searchsorted(node.vals, interval[0], side="left")
                higher = np.searchsorted(node.vals, interval[0], side="right")
            else:
                lower = np.searchsorted(node.vals, interval[0], side="left")
                higher = np.searchsorted(node.vals, interval[1], side="right")

            probs[i] += (higher - lower) / len(node.vals)

    return probs
