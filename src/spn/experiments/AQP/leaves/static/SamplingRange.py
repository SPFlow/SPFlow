"""
Created on May 22, 2018

@author: Moritz
"""

import numpy as np

from spn.experiments.AQP.Ranges import NumericRange


def sample_static_node(node, n_samples, rand_gen=None, ranges=None):

    if ranges is None or ranges[node.scope[0]] is None:
        return np.ones(n_samples) * node.val
    else:
        # Generate bins for the specified range
        rang = ranges[node.scope[0]]
        assert isinstance(rang, NumericRange)

        contained = False

        # Iterate over the specified ranges
        intervals = rang.get_ranges()
        for interval in intervals:

            lower = interval[0]
            higher = interval[0] if len(interval) == 1 else interval[1]

            if lower <= node.val and node.val <= higher:
                contained = True

    if contained:
        return np.ones(n_samples) * node.val
    else:
        samples = np.ones(n_samples)
        samples.fill(np.nan)
        return samples
