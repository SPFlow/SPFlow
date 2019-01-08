import bisect

import numpy as np

from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_ll
from spn.algorithms.Gradient import add_node_feature_gradient


def histogramm_gradient(node, input_vals=None, dtype=np.float64):
    if input_vals is None:
        raise ValueError("Input to piecewise_gradient cannot be None")
    data = input_vals

    breaks = node.breaks
    gradient = np.full(input_vals.shape, np.nan)

    nd = data[:, node.scope[0]]

    locs = np.searchsorted(breaks, nd)

    probs_left = histogram_ll(node.breaks, np.array(node.densities), locs - 1)
    probs_center = histogram_ll(node.breaks, np.array(node.densities), locs)
    probs_right = histogram_ll(node.breaks, np.array(node.densities), locs + 1)

    gradient[:, node.scope] = (((probs_center - probs_left) + probs_right - probs_center) / 2).reshape(-1, 1)

    return gradient


def add_histogram_gradient_support():
    add_node_feature_gradient(Histogram, histogramm_gradient)
