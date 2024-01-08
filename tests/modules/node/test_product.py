import unittest

from spflow.modules.node.leaf import Gaussian
from spflow.meta.data import Scope
from spflow import tensor as T
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.modules.node import ProductNode
from tests.fixtures import backend
import pytest


def make_product(num_inputs=2, children=None):
    if children is None:
        children = [make_leaf(scope=i) for i in range(num_inputs)]
    return ProductNode(children=children)


def make_leaf(mean=0.0, std=1.0, scope=0):
    return Gaussian(scope=Scope([scope]), mean=mean, std=std)


def make_data(mean=0.0, std=1.0, num_samples=10, dim=2):
    return T.randn(num_samples, dim) * std + mean


def test_log_likelihood(backend):
    product_node = make_product()
    data = make_data(dim=2)
    lls = log_likelihood(product_node, data)
    assert lls.shape == (data.shape[0], 1)


def test_sample(backend):
    product_node = make_product()
    samples = sample(product_node, num_samples=100)
    assert samples.shape == (100, 2)


def test_expectation_maximization():
    with T.backend_context(T.Backend.PYTORCH):
        product_node = make_product()
        data = make_data(dim=2)
        expectation_maximization(product_node, data, max_steps=10)


def test_constructor(backend):
    with pytest.raises(ValueError):
        # Product nodes must have disjunct input scopes
        make_product(children=[make_leaf(scope=0), make_leaf(scope=0)])


@pytest.mark.parametrize("prune", [True, False])
def test_marginalize(backend, prune):
    product_node = make_product()

    # Marginalize first scope
    marginalized_product_node = marginalize(product_node, [0], prune=prune)

    # Scope query should now only be [1]
    assert marginalized_product_node.scope.query == [1]

    if prune:
        # If pruning, Gaussian should be returned
        assert isinstance(marginalized_product_node, Gaussian)
    else:
        # Else ProductNode should be returned, with single child
        assert isinstance(marginalized_product_node, ProductNode)
        assert len(marginalized_product_node.children) == 1


if __name__ == "__main__":
    unittest.main()
