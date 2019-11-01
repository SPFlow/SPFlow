"""
Created on July 02, 2018

@author: Alejandro Molina
"""
import numpy as np

from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.histogram.Histograms import Histogram
import logging

from spn.structure.leaves.histogram.Inference import histogram_log_likelihood

logger = logging.getLogger(__name__)


def histogram_mode(node):
    areas = np.diff(node.breaks) * node.densities
    _x = np.argmax(areas)
    mode_value = node.bin_repr_points[_x]
    return mode_value


def histogram_bottom_up_log_ll(node, data=None, dtype=np.float64):
    probs = histogram_log_likelihood(node, data=data, dtype=dtype)
    mpe_ids = np.isnan(data[:, node.scope[0]])
    mode_data = np.ones((1, data.shape[1])) * histogram_mode(node)
    probs[mpe_ids] = histogram_log_likelihood(node, data=mode_data, dtype=dtype)

    return probs


def histogram_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=histogram_mode(node))


def add_histogram_mpe_support():
    add_node_mpe(Histogram, histogram_bottom_up_log_ll, histogram_top_down)
