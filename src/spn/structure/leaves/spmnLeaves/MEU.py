from spn.structure.leaves.histogram.MPE import histogram_mode
from spn.algorithms.MPE import get_mpe_top_down_leaf

from spn.algorithms.MPE import add_node_mpe

from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
import numpy as np

from spn.structure.leaves.spmnLeaves.Inference import utility_value


def utility_mode(node):
    return histogram_mode(node)

def utility_bottom_up_uVal(node, data=None, dtype=np.float64):
    uVal = utility_value(node, data=data, dtype=dtype)
    mpe_ids = np.isnan(data[:, node.scope[0]])
    mode_data = np.ones((1, data.shape[1])) * histogram_mode(node)
    uVal[mpe_ids] = utility_value(node, data=mode_data, dtype=dtype)

    return uVal

def utility_top_down(node, input_vals, lls_per_node, data=None):
    get_mpe_top_down_leaf(node, input_vals, data=data, mode=utility_mode(node))


def add_utility_mpe_support():
    add_node_mpe(Utility, utility_bottom_up_uVal, utility_top_down)