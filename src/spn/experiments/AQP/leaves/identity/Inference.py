'''
Created on June 21, 2018

@author: Moritz
'''

import numpy as np
from spn.experiments.AQP.leaves.static.StaticNumeric import StaticNumeric


def identity_likelihood(node, data, dtype=np.float64):
    assert len(node.scope) == 1, node.scope
    
    probs = np.zeros((data.shape[0], 1), dtype=dtype)
    nd = data[:, node.scope[0]]
    
    for i, val in enumerate(nd):
        if np.isnan(val):
            probs[i] = 1
        else:
            lower = np.searchsorted(node.vals, val, side='left')
            higher = np.searchsorted(node.vals, val, side='right')
            probs[i] = (higher-lower) / len(node.vals)
    
    return probs
    