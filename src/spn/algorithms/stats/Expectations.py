import numpy as np

from spn.algorithms.Inference import likelihood
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product, set_full_scope

_node_expectation = {}


def add_node_expectation(node_type, lambda_func):
    _node_expectation[node_type] = lambda_func


def prod_expectation(node, children, input_vals, moment=1, dtype=np.float64):
    assert node.full_scope is not None, 'Scope not set'
    llchildren = np.array(children).reshape(len(children), len(node.full_scope))
    ret_val = np.nansum(llchildren, axis=0, keepdims=True)
    assert len(ret_val.shape) == 2, 'Wrong return dimensionality'
    assert ret_val.shape[1] == len(node.full_scope), 'Wrong length of return array'
    return ret_val


def sum_expectation(node, children, input_vals, moment=1, dtype=np.float64):
    assert node.full_scope is not None, 'Scope not set'
    llchildren = np.array(children).reshape(len(children), len(node.full_scope))
    b = np.array(node.weights, dtype=dtype).reshape(1, -1)
    ret_val = np.dot(b, llchildren)
    assert len(ret_val.shape) == 2, 'Wrong return dimensionality'
    return ret_val


def Expectation(spn, feature_scope, evidence_scope, evidence, node_expectation=_node_expectation, moment=1):
    """Compute the Expectation:

        E[X_feature_scope | X_evidence_scope] given the spn and the evidence data

    Keyword arguments:
    spn -- the spn to compute the probabilities from
    feature_scope -- set() of integers, the scope of the features to get the expectation from
    evidence_scope -- set() of integers, the scope of the evidence features
    evidence -- numpy 2d array of the evidence data
    """
    if evidence_scope is None:
        evidence_scope = set()

    if not spn.full_scope:
        set_full_scope(spn)

    assert not (len(evidence_scope) > 0 and evidence is None)

    assert len(feature_scope.intersection(evidence_scope)) == 0

    marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)
    set_full_scope(marg_spn)

    def leaf_expectation(node, data, dtype=np.float64, **kwargs):
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((data.shape[0], data.shape[1]), dtype=dtype)
                exps[:] = node_expectation[t_node](node, moment=moment)
                return exps
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence)

    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(marg_spn, Leaf)}
    node_expectations.update({Sum: lambda x, c, input, dtype: sum_expectation(x, c, input, moment=moment, dtype=dtype),
                              Product: lambda x, c, input, dtype: prod_expectation(x, c, input, moment=moment, dtype=dtype)})

    if evidence is None:
        # fake_evidence is not used
        fake_evidence = np.zeros((1, len(spn.scope))).reshape(1,-1)
        expectation = likelihood(marg_spn, fake_evidence, node_likelihood=node_expectations)
        return expectation

    # if we have evidence, we want to compute the conditional expectation
    expectation = likelihood(marg_spn, evidence, node_likelihood=node_expectations)
    expectation = expectation / likelihood(marginalize(marg_spn, keep=evidence_scope), evidence)

    return expectation


def get_means(spn):
    if not spn.full_scope:
        set_full_scope(spn)
    return Expectation(spn, set(spn.full_scope), None, None)


def get_variances(spn):
    if not spn.full_scope:
        set_full_scope(spn)
    return Expectation(spn, set(spn.full_scope), None, None,
                       moment=2) - get_means(spn) ** 2
