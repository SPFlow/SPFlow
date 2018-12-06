import bisect

import numpy as np

from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_ll


def histogramm_gradient(node, input_vals=None, dtype=np.float64):
    if input_vals is None:
        raise ValueError("Input to piecewise_gradient cannot be None")
    data = input_vals

    breaks = node.breaks

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    locs = np.searchsorted(breaks, nd)

    probs_left = histogram_ll(node.breaks, np.array(node.densities), locs - 1)
    probs_center = histogram_ll(node.breaks, np.array(node.densities), locs)
    probs_right = histogram_ll(node.breaks, np.array(node.densities), locs + 1)

    gradients = ((probs_center - probs_left) + probs_right - probs_center) / 2
    gradients[marg_ids] = np.nan

    return gradients.reshape((-1, 1))


def add_histogram_gradient_support():
    add_node_gradients(Histogram, histogramm_gradient)
