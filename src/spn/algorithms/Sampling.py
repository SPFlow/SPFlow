'''
Created on April 5, 2018

@author: Alejandro Molina
'''
import logging

import numpy as np

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.io.Text import str_to_spn, to_JSON
from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Parametric
from spn.structure.leaves.parametric.Sampling import sample_parametric_node



def validate_ids(node):
    all_nodes = get_nodes_by_type(node)

    ids = set()
    for n in all_nodes:
        ids.add(n.id)

    assert len(ids) == len(all_nodes), "not all nodes have ID's"

    assert min(ids) == 0 and max(ids) == len(ids) - 1, "ID's are not in order"


def reset_node_counters(node):
    all_nodes = get_nodes_by_type(node)
    max_id = 0
    for n in all_nodes:
        # reset sum node counts
        if isinstance(n, Sum):
            n.edge_counts = np.zeros(len(n.children), dtype=np.int64)
        # sets nr_nodes to the max id
        max_id = max(max_id, n.id)
        n.row_ids = []
    return max_id


def sample_instances(node, D, n_samples, rand_gen, return_Zs=True, return_partition=True, dtype=np.float64):
    """
    Implementing hierarchical sampling

    D could be extracted by traversing node
    """

    sum_nodes = get_nodes_by_type(node, Sum)
    n_sum_nodes = len(sum_nodes)

    if return_Zs:
        Z = np.zeros((n_samples, n_sum_nodes), dtype=np.int64)
        Z_id_map = {}
        for j, s in enumerate(sum_nodes):
            Z_id_map[s.id] = j

    if return_partition:
        P = np.zeros((n_samples, D), dtype=np.int64)

    instance_ids = np.arange(n_samples)
    X = np.zeros((n_samples, D), dtype=dtype)

    _max_id = reset_node_counters(node)

    def _sample_instances(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_instances(c, row_ids)
            return

        if isinstance(node, Sum):
            w_children_log_probs = np.zeros((len(row_ids), len(node.weights)))
            for i, c in enumerate(node.children):
                w_children_log_probs[:, i] = np.log(node.weights[i])

            z_gumbels = rand_gen.gumbel(loc=0, scale=1,
                                        size=(w_children_log_probs.shape[0], w_children_log_probs.shape[1]))
            g_children_log_probs = w_children_log_probs + z_gumbels
            rand_child_branches = np.argmax(g_children_log_probs, axis=1)

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_instances(c, new_row_ids)

                if return_Zs:
                    Z[new_row_ids, Z_id_map[node.id]] = i

        if isinstance(node, Leaf):
            #
            # sample from leaf
            X[row_ids, node.scope] = sample_parametric_node(
                node, n_samples=len(row_ids), rand_gen=rand_gen)
            if return_partition:
                P[row_ids, node.scope] = node.id

            return

    _sample_instances(node, instance_ids)

    if return_Zs:
        if return_partition:
            return X, Z, P

        return X, Z

    if return_partition:
        return X, P

    return X

