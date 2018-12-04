"""
Created on May 23, 2018

@author: Moritz
"""

import numpy as np


from spn.structure.Base import Product, Sum, Leaf, get_nodes_by_type
from spn.algorithms.Inference import _node_likelihood


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


def set_weights_for_evidence(node, ranges, dtype=np.float64, node_likelihood=_node_likelihood):

    if isinstance(node, Product):

        prob = 1.0
        for c in node.children:
            prob *= set_weights_for_evidence(c, ranges, node_likelihood=node_likelihood)

        return prob

    elif isinstance(node, Sum):
        evidence_weights = np.zeros(len(node.children))

        for i, c in enumerate(node.children):
            prob = set_weights_for_evidence(c, ranges, node_likelihood=node_likelihood)
            evidence_weights[i] = prob * node.weights[i]

        if np.sum(evidence_weights) == 0:
            return 0

        node.evidence_weights = evidence_weights / np.sum(evidence_weights)
        return np.sum(evidence_weights)

    elif isinstance(node, Leaf):
        t_node = type(node)
        if t_node in node_likelihood:
            ranges = np.array([ranges])
            return node_likelihood[t_node](node, ranges, dtype=dtype, node_likelihood=node_likelihood)
        else:
            raise Exception("No log-likelihood method specified for node type: " + str(type(node)))


def sample_instances(
    node, D, n_samples, rand_gen, ranges=None, dtype=np.float64, node_sample=None, node_likelihood=_node_likelihood
):

    instance_ids = np.arange(n_samples)
    X = np.zeros((n_samples, D), dtype=dtype)

    _max_id = reset_node_counters(node)
    result = set_weights_for_evidence(node, ranges, node_likelihood=node_likelihood)

    if result == 0:
        return np.zeros((0, D), dtype=dtype)

    def _sample_instances(node, row_ids):
        if len(row_ids) == 0:
            return
        node.row_ids = row_ids

        if isinstance(node, Product):
            for c in node.children:
                _sample_instances(c, row_ids)
            return

        if isinstance(node, Sum):

            rand_child_branches = np.random.choice(
                np.arange(len(node.evidence_weights)), p=node.evidence_weights, size=len(row_ids)
            )

            for i, c in enumerate(node.children):
                new_row_ids = row_ids[rand_child_branches == i]
                node.edge_counts[i] = len(new_row_ids)
                _sample_instances(c, new_row_ids)

        if isinstance(node, Leaf):

            t_node = type(node)
            if t_node in node_sample:
                X[row_ids, node.scope] = node_sample[t_node](node, len(row_ids), rand_gen, ranges)
            else:
                raise Exception("No sample method specified for node type: " + str(type(node)))

            return

    _sample_instances(node, instance_ids)

    return X


if __name__ == "__main__":

    import os

    os.environ["R_HOME"] = "C:/Program Files/R/R-3.3.3"
    os.environ["R_USER"] = "Moritz"

    from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
    from spn.structure.leaves.piecewise.SamplingRange import sample_piecewise_node
    from spn.structure.leaves.piecewise.InferenceRange import piecewise_likelihood_range

    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.leaves.parametric.SamplingRange import sample_categorical_node
    from spn.structure.leaves.parametric.InferenceRange import categorical_likelihood_range

    from spn.structure.Base import Context
    from spn.structure.StatisticalTypes import MetaType
    from spn.experiments.AQP.Ranges import NominalRange, NumericRange

    from spn.algorithms import SamplingRange

    rand_gen = np.random.RandomState(100)

    # Create SPN
    node1 = Categorical(p=[0.9, 0.1], scope=[0])
    node2 = Categorical(p=[0.1, 0.9], scope=[0])

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 10.0, 0.0, 0.0, 0.0]
    x, y = np.array(x), np.array(y)
    auc = np.trapz(y, x)
    y = y / auc
    node3 = PiecewiseLinear(x_range=x, y_range=y, bin_repr_points=x[1:-1], scope=[1])

    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 0.0, 0.0, 10.0, 0.0]
    x, y = np.array(x), np.array(y)
    auc = np.trapz(y, x)
    y = y / auc
    node4 = PiecewiseLinear(x_range=x, y_range=y, bin_repr_points=x[1:-1], scope=[1])

    root_node = 0.49 * (node1 * node3) + 0.51 * (node2 * node4)

    # Set context
    # meta_types = [MetaType.DISCRETE, MetaType.REAL]
    # domains = [[0,1],[0.,4.]]
    # ds_context = Context(meta_types=meta_types, domains=domains)

    inference_support_ranges = {PiecewiseLinear: piecewise_likelihood_range, Categorical: categorical_likelihood_range}

    node_sample = {Categorical: sample_categorical_node, PiecewiseLinear: sample_piecewise_node}

    ranges = [NominalRange([0]), None]
    samples = SamplingRange.sample_instances(
        root_node, 2, 30, rand_gen, ranges=ranges, node_sample=node_sample, node_likelihood=inference_support_ranges
    )  # , return_Zs, return_partition, dtype)
    print("Samples: " + str(samples))

    ranges = [NominalRange([0]), NumericRange([[3.0, 3.1], [3.5, 4.0]])]
    samples = SamplingRange.sample_instances(
        root_node, 2, 30, rand_gen, ranges=ranges, node_sample=node_sample, node_likelihood=inference_support_ranges
    )  # , return_Zs, return_partition, dtype)
    print("Samples: " + str(samples))
