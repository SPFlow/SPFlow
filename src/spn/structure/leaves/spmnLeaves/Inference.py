
import numpy as np

from spn.algorithms.Inference import EPSILON, add_node_likelihood

from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
from spn.structure.leaves.histogram.Inference import histogram_likelihood

def utility_value(node, data=None, dtype=np.float64):
    uVal = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    uVal[~marg_ids] = nd[~marg_ids].reshape((-1,1))

    uVal[uVal < EPSILON] = EPSILON

    return uVal


def add_utility_inference_support():
    add_node_likelihood(Utility, histogram_likelihood)
