"""
Created on October 19, 2018

@author: Nicola Di Mauro
"""

from spn.structure.leaves.cltree.CLTree import CLTree

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
import numpy as np


def compute_cooccurences(data, C, NZ, r, c):
    for k in range(r):
        non_zeros = 0
        for i in range(c):
            if data[k, i]:
                NZ[non_zeros] = i
                non_zeros += 1
                for j in range(non_zeros):
                    v = NZ[j]
                    C[v, i] += 1
    for i in range(1, c):
        for j in range(i):
            C[i, j] = C[j, i]


def compute_log_probs(node, data, alpha):
    log_probs = np.zeros((node.n_features, 2))
    log_j_probs = np.zeros((node.n_features, node.n_features, 2, 2))

    cooccurences = np.zeros((data.shape[1], data.shape[1]))
    NZ = np.zeros(data.shape[1], dtype="int")
    compute_cooccurences(data, cooccurences, NZ, data.shape[0], data.shape[1])
    p = cooccurences.diagonal()

    n_samples = data.shape[0]

    for i in range(node.n_features):
        prob = (p[i] + 2 * alpha) / (n_samples + 4 * alpha)
        log_probs[i, 0] = np.log(1 - prob)
        log_probs[i, 1] = np.log(prob)

    for i in range(node.n_features):
        for j in range(node.n_features):
            log_j_probs[i, j, 1, 1] = np.log((cooccurences[i, j] + alpha) / (n_samples + 4 * alpha))
            log_j_probs[i, j, 0, 1] = np.log(
                (cooccurences[j, j] - cooccurences[i, j] + alpha) / (n_samples + 4 * alpha)
            )
            log_j_probs[i, j, 1, 0] = np.log(
                (cooccurences[i, i] - cooccurences[i, j] + alpha) / (n_samples + 4 * alpha)
            )
            log_j_probs[i, j, 0, 0] = np.log(
                (n_samples - cooccurences[j, j] - cooccurences[i, i] + cooccurences[i, j] + alpha)
                / (n_samples + 4 * alpha)
            )
            log_j_probs[j, i, 1, 1] = log_j_probs[i, j, 1, 1]
            log_j_probs[j, i, 1, 0] = log_j_probs[i, j, 0, 1]
            log_j_probs[j, i, 0, 1] = log_j_probs[i, j, 1, 0]
            log_j_probs[j, i, 0, 0] = log_j_probs[i, j, 0, 0]

    return (log_probs, log_j_probs)


def update_cltree_parameters_mle(node, data, alpha=0.01):
    """ learn the structure and parameters of a CLTree """

    log_factors = np.zeros((node.n_features, 2, 2))

    if node.n_features == 1:
        p = (data.sum() + 2 * alpha) / (len(data) + 4 * alpha)

        log_factors[0, 0, 0] = np.log(1 - p)
        log_factors[0, 0, 1] = np.log(1 - p)
        log_factors[0, 1, 0] = np.log(p)
        log_factors[0, 1, 1] = np.log(p)

        node.tree = [-1]
        node.df_order = [0]
        node.post_order = [0]

    else:
        node.tree = [0] * node.n_features
        node.tree[0] = -1

        (log_probs, log_j_probs) = compute_log_probs(node, data, alpha)

        MI = np.zeros((node.n_features, node.n_features))
        for i in range(node.n_features):
            for j in range(i + 1, node.n_features):
                for v0 in range(2):
                    for v1 in range(2):
                        MI[i, j] = MI[i, j] + np.exp(log_j_probs[i, j, v0, v1]) * (
                            log_j_probs[i, j, v0, v1] - log_probs[i, v0] - log_probs[j, v1]
                        )
                MI[j, i] = MI[i, j]

        mst = minimum_spanning_tree(-(MI))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        node.df_order = dfs_tree[0].tolist()
        node.post_order = dfs_tree[0][::-1].tolist()

        for p in range(1, node.n_features):
            node.tree[p] = dfs_tree[1][p]

        # computing the factored represetation

        for feature in range(0, node.n_features):
            if node.tree[feature] == -1:
                log_factors[feature, 0, 0] = log_probs[feature, 0]
                log_factors[feature, 0, 1] = log_probs[feature, 0]
                log_factors[feature, 1, 0] = log_probs[feature, 1]
                log_factors[feature, 1, 1] = log_probs[feature, 1]
            else:
                parent = int(node.tree[feature])
                for feature_val in range(2):
                    for parent_val in range(2):
                        log_factors[feature, feature_val, parent_val] = (
                            log_j_probs[feature, parent, feature_val, parent_val] - log_probs[parent, parent_val]
                        )

    node.log_factors = log_factors.tolist()
