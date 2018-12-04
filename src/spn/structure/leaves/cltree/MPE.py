"""
Created on October 22, 2018

@author: Nicola DI Mauro
"""
from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
from spn.structure.leaves.cltree.Inference import cltree_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree

import numpy as np


def cltree_bottom_up_ll(node, data, dtype=np.float64):
    probs = np.ones((data.shape[0], 1))
    cltree_mpe(node, data, probs)
    return probs


def cltree_top_down(node, input_vals, data, lls_per_node=None, dtype=np.float64):
    return


def add_cltree_mpe_support():
    add_node_mpe(CLTree, cltree_bottom_up_ll, cltree_top_down)


def cltree_mpe(node, data, probs):

    log_factors = np.array(node.log_factors)

    for r in range(data.shape[0]):

        messages = np.zeros((node.n_features, 2))
        states = [[0, 0] for i in range(node.n_features)]
        MAP = {}

        for i in node.post_order:
            if i != 0:
                state_evidence = data[r, node.scope[i]]
                if not np.isnan(state_evidence):
                    state_evidence = int(state_evidence)
                    states[i][0] = state_evidence
                    states[i][1] = state_evidence
                    messages[node.tree[i], 0] += log_factors[i, state_evidence, 0] + messages[i, state_evidence]
                    messages[node.tree[i], 1] += log_factors[i, state_evidence, 1] + messages[i, state_evidence]
                else:
                    state_evidence_parent = data[r, node.scope[node.tree[i]]]
                    if not np.isnan(state_evidence_parent):
                        state_evidence_parent = int(state_evidence_parent)
                        if (
                            log_factors[i, 0, state_evidence_parent] + messages[i, 0]
                            > log_factors[i, 1, state_evidence_parent] + messages[i, 1]
                        ):
                            states[i][state_evidence_parent] = 0
                            messages[node.tree[i], state_evidence_parent] += (
                                log_factors[i, 0, state_evidence_parent] + messages[i, 0]
                            )
                        else:
                            states[i][state_evidence_parent] = 1
                            messages[node.tree[i], state_evidence_parent] += (
                                log_factors[i, 1, state_evidence_parent] + messages[i, 1]
                            )
                    else:
                        for parent in range(2):
                            if log_factors[i, 0, parent] + messages[i, 0] > log_factors[i, 1, parent] + messages[i, 1]:
                                states[i][parent] = 0
                                messages[node.tree[i], parent] += log_factors[i, 0, parent] + messages[i, 0]
                            else:
                                states[i][parent] = 1
                                messages[node.tree[i], parent] += log_factors[i, 1, parent] + messages[i, 1]
        logprob = 0.0
        for i in node.df_order:
            if node.tree[i] == -1:
                state_evidence = data[r, node.scope[i]]
                if not np.isnan(state_evidence):
                    state_evidence = int(state_evidence)
                    MAP[i] = state_evidence
                    logprob += log_factors[i, int(MAP[i]), 0]
                else:
                    if log_factors[i, 0, 0] + messages[i, 0] > log_factors[i, 1, 0] + messages[i, 1]:
                        MAP[i] = 0
                        data[r, node.scope[i]] = 0
                    else:
                        MAP[i] = 1
                        data[r, node.scope[i]] = 1
                    logprob += log_factors[i, int(MAP[i]), 0]
            else:
                MAP[i] = states[i][MAP[node.tree[i]]]
                if np.isnan(data[r, node.scope[i]]):
                    data[r, node.scope[i]] = MAP[i]
                logprob += log_factors[i, int(MAP[i]), int(MAP[node.tree[i]])]

        probs[r] = np.exp(logprob)
