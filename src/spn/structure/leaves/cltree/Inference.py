'''
Created on October 19, 2018

@author: Nicola Di Mauro
'''

from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree

import numpy as np

def cltree_likelihood(node, data=None, dtype=np.float64):
    probs = np.zeros(data.shape[0], dtype=dtype)

    for feature in range(0, node.n_features):
        parent = node.tree[feature]
        if parent == -1:
            probs = probs + node.log_factors[feature, data[:,node.scope[feature]],0]
        else:
            probs = probs + node.log_factors[feature, data[:,node.scope[feature]], data[:,node.scope[parent]]]

    return np.exp(probs.reshape(data.shape[0],1))
    


def add_cltree_inference_support():
    add_node_likelihood(CLTree, cltree_likelihood)
