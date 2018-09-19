import numpy as np

from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type, eval_spn_bottom_up, set_full_scope, assign_ids

from spn.algorithms.TransformStructure import Copy, Prune
from spn.algorithms.Inference import log_likelihood


def prod_condition(node, children, evidence, scope=None):
    if not scope.intersection(node.scope):
        return Copy(node), 0
    new_node = Product()
    new_node.scope = list(set(node.scope) - scope)
    probability = 0
    for c in children:
        if c[0]:
            new_node.children.append(c[0])
        probability += float(c[1])
    return new_node, probability


def sum_condition(node, children, evidence, scope=None):
    if not scope.intersection(node.scope):
        return Copy(node), 0
    new_node = Sum()
    new_node.scope = list(set(node.scope) - scope)
    new_weights = []
    for i, c in enumerate(children):
        if c[0]:
            new_node.children.append(c[0])
            new_weights.append(node.weights[i] * np.exp(c[1]))
    new_node.weights = [w/sum(new_weights) for w in new_weights]
    return new_node, np.log(sum(new_weights))


def leaf_condition(node, evidence, scope=None):
    if not scope.intersection(node.scope):
        return Copy(node), 0

    _likelihood = log_likelihood(node, evidence)
    return None, _likelihood


def condition(spn, evidence):
    if not spn.full_scope:
        set_full_scope(spn)
    scope = set([i for i in range(len(spn.full_scope)) if not np.isnan(evidence)[0][i]])
    node_conditions = {type(leaf): leaf_condition for leaf in get_nodes_by_type(spn, Leaf)}
    node_conditions.update({Sum: sum_condition,
                            Product: prod_condition})

    new_root, val = eval_spn_bottom_up(spn, node_conditions, input_vals=evidence, scope=scope)
    assign_ids(new_root)
    return Prune(new_root)
