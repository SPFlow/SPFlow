"""
Created on October 22, 2018

@author: Nicola Di Mauro
@author: Antonio Vergari
"""
from spn.algorithms.Sampling import add_node_sampling
from spn.structure.leaves.cltree.CLTree import CLTree


def sample_cltree_node(node, n_samples, data, rand_gen):

    X = np.array((n_samples, node.n_features))
    sample = np.zeros(node.n_features)
    log_factors = np.array(node.log_factors)
    # forward sampling
    topological_order = [0] * node.n_features
    visited = {}
    to_visit = {}
    visited.add(0)
    for i in range(1, node.n_features):
        to_visit.add(i)
    i = 1
    while to_visit:
        for ntv in to_visit:
            if not node.tree[ntv] in visited:
                topological_order[i] = ntv
                to_visit.remove(ntv)
                visited.add(ntv)
                i += 1
                break

    for s in range(n_samples):

        # sampling the root of the tree
        sample[0] = np.random.binomial(np.exp(log_factors[0][1][0]))

        for i in range(1, node.n_features):
            feature_to_sample = topological_order[i]
            parent = node.tree[feature_to_sample]
            parent_sampled = sample[parent]
            sample[feature_to_sample] = np.random.binomial(np.exp(log_factors[feature_to_sample][1][parent_sampled]))

        X[s] = sample
    return X


def add_cltree_sampling_support():
    add_node_sampling(CLTree, sample_cltree_node)
