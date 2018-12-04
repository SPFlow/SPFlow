"""
Created on June 21, 2018

@author: Moritz
"""

import numpy as np

from spn.algorithms.Inference import add_node_likelihood
from spn.experiments.AQP.leaves.static.StaticNumeric import StaticNumeric


def static_likelihood(node, data, dtype=np.float64):
    assert len(node.scope) == 1, node.scope
    assert data.shape[1] == 1, data.shape

    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    nd = data[:, node.scope[0]]

    probs[((np.isnan(nd)) | (nd == node.val))] = 1

    return probs


def add_static_inference_support():
    add_node_likelihood(StaticNumeric, static_likelihood)
