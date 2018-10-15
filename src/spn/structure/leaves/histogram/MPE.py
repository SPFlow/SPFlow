'''
Created on July 02, 2018

@author: Alejandro Molina
'''
import numpy as np

from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_likelihood


def histogram_mode(node):
    areas = np.diff(node.breaks) * node.densities
    _x = np.argmax(areas)
    mode_value = node.bin_repr_points[_x]
    return mode_value


def histogram_bottom_up_ll(node, data=None, dtype=np.float64):
    probs = histogram_likelihood(node, data=data, dtype=dtype)

    mpe_ids = np.isnan(data[:, node.scope[0]])
    probs[mpe_ids] = histogram_mode(node)

    return probs


def histogram_top_down(node, input_vals, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=histogram_mode(node))


def add_histogram_mpe_support():
    add_node_mpe(Histogram, histogram_bottom_up_ll, histogram_top_down)
