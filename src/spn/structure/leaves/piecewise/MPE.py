"""
Created on October 24, 2018


@author: Claas Voelcker
"""
import numpy as np

from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
from spn.structure.leaves.piecewise.Inference import piecewise_likelihood


def piecewise_mode(node):
    _x = np.argmax(node.y_range)
    mode_value = node.x_range[_x]
    return mode_value


def piecewise_bottom_up_ll(node, data=None, dtype=np.float64):
    probs = piecewise_likelihood(node, data=data, dtype=dtype)
    mpe_ids = np.isnan(data[:, node.scope[0]])
    probs[mpe_ids] = piecewise_mode(node)

    return probs


def piecewise_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=piecewise_mode(node))


def add_piecewise_mpe_support():
    add_node_mpe(PiecewiseLinear, piecewise_bottom_up_ll, piecewise_top_down)
