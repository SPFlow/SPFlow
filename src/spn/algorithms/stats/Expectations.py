from spn.algorithms.Inference import likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product
import numpy as np

_node_expectation = {}


def add_node_expectation(node_type, lambda_func):
    _node_expectation[node_type] = lambda_func


def Expectation(spn, feature_scope, evidence_scope, evidence, node_expectation=_node_expectation):
    if evidence is None:
        evidence = np.zeros((1, 1)).reshape(1,1)

    if evidence_scope is None:
        evidence_scope = set()

    marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)

    def leaf_expectation(node, data, dtype=np.float64, **kwargs):
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((data.shape[0], 1), dtype=dtype)
                exps[:] = node_expectation[t_node](node)
                return exps
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence)

    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(marg_spn, Leaf)}
    node_expectations.update({Sum: sum_likelihood, Product: prod_likelihood})

    expectation = likelihood(marg_spn, evidence, node_likelihood=node_expectations)

    return expectation
