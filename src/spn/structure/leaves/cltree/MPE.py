'''
Created on October 22, 2018

@author: Nicola DI Mauro
'''
from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.cltree.Inference import cltree_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree

import numpy as np

def get_cltree_bottom_up_ll(ll_func):
    return 0


def get_cltree_top_down_ll():
    return 0


def add_cltree_mpe_support():
    add_node_mpe(CLTree, get_cltree_bottom_up_ll(cltree_likelihood),
                 get_cltree_top_down_ll())

def cltree_mpe(node, data):
    messages = np.zeros((node.n_features, 2))
    states = [ [0,0] for i in range(node.n_features) ] 
    MAP = {}

    for i in node.post_order:
        if i != 0:
            state_evidence = data[node.scope[i]]
            if state_evidence != np.nan:
                states[i][0] = state_evidence
                states[i][1] = state_evidence
                messages[node.tree[i],0]+= node.log_factors[i,state_evidence,0]+messages[i,state_evidence]
                messages[node.tree[i],1]+= node.log_factors[i,state_evidence,1]+messages[i,state_evidence]
            else:
                state_evidence_parent = data[node.scope[node.tree[i]]]
                if state_evidence_parent != np.nan:
                    if (node.log_factors[i,0,state_evidence_parent] + messages[i,0] > node.log_factors[i,1,state_evidence_parent] + messages[i,1]):
                        states[i][state_evidence_parent] = 0
                        messages[node.tree[i],state_evidence_parent]+= node.log_factors[i,0,state_evidence_parent]+messages[i,0]
                    else:
                        states[i][state_evidence_parent] = 1
                        messages[node.tree[i],state_evidence_parent]+= node.log_factors[i,1,state_evidence_parent]+messages[i,1]
                else:
                    for parent in range(2):
                        if (node.log_factors[i,0,parent] + messages[i,0] > node.log_factors[i,1,parent] + messages[i,1]):
                            states[i][parent] = 0
                            messages[node.tree[i],parent]+= node.log_factors[i,0,parent]+messages[i,0]
                        else:
                            states[i][parent] = 1
                            messages[node.tree[i],parent]+= node.log_factors[i,1,parent]+messages[i,1]
    logprob = 0.0
    for i in node.df_order:
        if node.tree[i] == -1:
            state_evidence = data[scope[i]]
            if state_evidence != np.nan:
                MAP[node.scope[i]] = state_evidence
                logprob += node.log_factors[i,MAP[node.scope[i]],0]
            else:
                if node.log_factors[i,0,0] + messages[i,0] > node.log_factors[i,1,0]+messages[i,1]:
                    MAP[node.scope[i]] = 0
                else:
                    MAP[node.scope[i]] = 1
                    logprob += node.log_factors[i,MAP[node.scope[i]],0]
        else:
            MAP[node.scope[i]] = states[i][MAP[node.scope[node.tree[i]]]]
            logprob += node.log_factors[i,MAP[node.scope[i]],MAP[node.scope[node.tree[i]]]]
        return (MAP, logprob)


