"""
Created on October 19, 2018

@author: Nicola Di Mauro
"""

from spn.algorithms.Inference import add_node_likelihood
from spn.structure.leaves.cltree.CLTree import CLTree

import numpy as np
import logging

logger = logging.getLogger(__name__)


def cltree_log_likelihood(node, data=None, dtype=np.float64, **kwargs):
    log_probs = np.zeros(data.shape[0], dtype=dtype)
    log_factors = np.array(node.log_factors)

    if np.isnan(np.sum(data)):

        for r in range(data.shape[0]):

            messages = np.zeros((node.n_features, 2))
            logprob = 0.0
            for i in node.post_order:
                state_evidence = data[r, node.scope[i]]
                if i != 0:
                    if not np.isnan(state_evidence):
                        messages[node.tree[i], 0] += (
                            log_factors[i, int(state_evidence), 0] + messages[i, int(state_evidence)]
                        )
                        messages[node.tree[i], 1] += (
                            log_factors[i, int(state_evidence), 1] + messages[i, int(state_evidence)]
                        )
                    else:
                        # marginalization
                        messages[node.tree[i], 0] += np.log(
                            np.exp(log_factors[i, 0, 0] + messages[i, 0])
                            + np.exp(log_factors[i, 1, 0] + messages[i, 1])
                        )
                        messages[node.tree[i], 1] += np.log(
                            np.exp(log_factors[i, 0, 1] + messages[i, 0])
                            + np.exp(log_factors[i, 1, 1] + messages[i, 1])
                        )
                else:
                    if not np.isnan(state_evidence):
                        logprob = log_factors[i, int(state_evidence), 0] + messages[0, int(state_evidence)]
                    else:
                        # marginalization
                        logprob = np.log(
                            np.exp(log_factors[i, 0, 0] + messages[0, 0])
                            + np.exp(log_factors[i, 1, 0] + messages[0, 1])
                        )
            log_probs[r] = logprob

    else:

        for feature in range(0, node.n_features):
            parent = node.tree[feature]
            if parent == -1:
                log_probs = log_probs + log_factors[feature, data[:, node.scope[feature]], 0]
            else:
                log_probs = log_probs + log_factors[feature, data[:, node.scope[feature]], data[:, node.scope[parent]]]

    return log_probs.reshape(data.shape[0], 1)


def add_cltree_inference_support():
    add_node_likelihood(CLTree, log_lambda_func=cltree_log_likelihood)
