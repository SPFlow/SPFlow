from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Leaf, get_nodes_by_type
import numpy as np

_node_expectation = {}


def add_node_expectation(node_type, lambda_func):
    _node_expectation[node_type] = lambda_func


def Expectation(spn, feature_scope, evidence_scope, evidence, ds_context, node_expectation=_node_expectation):
    if evidence_scope is None:
        evidence_scope = set()

    marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)

    def leaf_expectation(node, data, dtype=np.float64, node_log_likelihood=None):
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((data.shape[0], 1), dtype=dtype)
                exps[:] = node_expectation[t_node](node, ds_context)
                return np.log(exps)
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return log_likelihood(node, evidence)

    node_log_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(marg_spn, Leaf)}

    log_expectation = log_likelihood(marg_spn, evidence, node_log_likelihood=node_log_expectations)

    return np.exp(log_expectation)
