from spn.algorithms.Inference import likelihood, sum_likelihood, prod_likelihood
from spn.structure.Base import Leaf, get_nodes_by_type, Sum, Product
import numpy as np

_node_expectation = {}


def add_node_expectation(node_type, lambda_func):
    _node_expectation[node_type] = lambda_func


def Expectation(marg_spn, feature_scope, evidence_scope, evidence, node_expectation=_node_expectation):

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

    assert not (len(evidence_scope) > 0 and evidence is None)

    assert len(feature_scope.intersection(evidence_scope)) == 0


    #marg_spn = marginalize(spn, keep=feature_scope | evidence_scope)

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

    if evidence is None:
        #fake_evidence is not used
        fake_evidence = np.zeros((1, len(marg_spn.scope))).reshape(1,-1)
        expectation = likelihood(marg_spn, fake_evidence, node_likelihood=node_expectations)
        return expectation

    #if we have evidence, we want to compute the conditional expectation
    expectation = likelihood(marg_spn, evidence, node_likelihood=node_expectations)
    expectation = expectation / likelihood(marg_spn, evidence)

    return expectation
