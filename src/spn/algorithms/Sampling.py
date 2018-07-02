'''
Created on April 5, 2018

@author: Alejandro Molina
'''
import logging

import numpy as np

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.algorithms.Validity import is_valid
from spn.io.Text import str_to_spn, to_JSON
from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type, eval_spn_bottom_up
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.parametric.Parametric import Parametric
from spn.structure.leaves.parametric.Sampling import sample_parametric_node


def sample_instances(node, data, rand_gen):
    """
    Implementing hierarchical sampling

    """

    # first, we do a bottom-up pass to compute the likelihood taking into account marginals.
    # then we do a top-down pass, to sample taking into account the likelihoods.

    valid, err = is_valid(node)
    assert valid, err

    assert np.all(
        np.any(np.isnan(data), axis=1)), "each row must have at least a nan value where the samples will be substituted"

    nodes = get_nodes_by_type(node)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    log_likelihood(node, data, dtype=data.dtype, lls_matrix=lls_per_node)

    instance_ids = np.arange(data.shape[0])

    eval_spn_bottom_up(node, node_likelihood, input_vals=data, validation_function=val_funct,
                       dtype=dtype)

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
