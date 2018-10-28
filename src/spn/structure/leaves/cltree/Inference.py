'''
Created on October 19, 2018

@author: Nicola Di Mauro
'''

from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree

import numpy as np

def cltree_likelihood_(node, data=None, dtype=np.float64):
    probs = np.zeros(data.shape[0], dtype=dtype)

    for feature in range(0, node.n_features):
        parent = node.tree[feature]
        if parent == -1:
            probs = probs + node.log_factors[feature,
                                             data[:,node.scope[feature]],0]
        else:
            probs = probs + node.log_factors[feature,
                                             data[:,node.scope[feature]],
                                             data[:,node.scope[parent]]]

    return np.exp(probs.reshape(data.shape[0],1))
    

def cltree_likelihood(node, data=None, dtype=np.float64):
    probs = np.zeros(data.shape[0], dtype=dtype)

    for r in range(data.shape[0]):

        messages = np.zeros((node.n_features, 2))
        logprob = 0.0
        for i in node.post_order:
            state_evidence = data[r,node.scope[i]]
            if i != 0:
                if not np.isnan(state_evidence):
                    messages[node.tree[i],0] += \
                                                node.log_factors[i,int(state_evidence),0] + \
                                                messages[i,int(state_evidence)]
                    messages[node.tree[i],1] += \
                                                node.log_factors[i,int(state_evidence),1] + \
                                                messages[i,int(state_evidence)]
                else:
                    # marginalization
                    messages[node.tree[i], 0] += \
                                                 np.log(np.exp(node.log_factors[i, 0, 0] + messages[i,0]) +
                                                        np.exp(node.log_factors[i, 1, 0] + messages[i,1]))
                    messages[node.tree[i], 1] += \
                                                 np.log(np.exp(node.log_factors[i, 0, 1] + messages[i,0]) +
                                                        np.exp(node.log_factors[i, 1, 1] + messages[i,1]))
            else:
                if not np.isnan(state_evidence):
                    logprob = node.log_factors[i, int(state_evidence),0] + \
                              messages[0, int(state_evidence)]
                else:
                    # marginalization
                    logprob = \
                              np.log( np.exp(node.log_factors[i,0,0] + messages[0,0]) +
                                      np.exp(node.log_factors[i,1,0]+messages[0,1]))
        probs[r] = logprob
    return np.exp(probs.reshape(data.shape[0],1))


def add_cltree_inference_support():
    add_node_likelihood(CLTree, cltree_likelihood)
