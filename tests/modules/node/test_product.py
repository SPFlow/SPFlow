import unittest

from spflow.modules.node.leaf import Normal
from spflow.meta.data import Scope
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.modules.node import ProductNode
import pytest
import torch


def make_product(num_inputs=2, inputs=None):
    if inputs is None:
        inputs = [make_leaf(scope=i) for i in range(num_inputs)]
    return ProductNode(inputs=inputs)


def make_leaf(mean=0.0, std=1.0, scope=0):
    return Normal(scope=Scope([scope]), mean=mean, std=std)


def make_data(mean=0.0, std=1.0, num_samples=10, dim=2):
    return torch.randn(num_samples, dim) * std + mean


def test_log_likelihood():
    product_node = make_product()
    data = make_data(dim=2)
    lls = log_likelihood(product_node, data)
    assert lls.shape == (data.shape[0], 1)


def test_sample():
    product_node = make_product()
    samples = sample(product_node, num_samples=100)
    assert samples.shape == (100, 2)


def test_expectation_maximization():
    product_node = make_product()
    data = make_data(dim=2)
    expectation_maximization(product_node, data, max_steps=10)


def test_constructor():
    with pytest.raises(ValueError):
        # Product nodes must have disjunct input scopes
        make_product(inputs=[make_leaf(scope=0), make_leaf(scope=0)])


@pytest.mark.parametrize("prune", [True, False])
def test_marginalize(prune):
    product_node = make_product()

    # Marginalize first scope
    marginalized_product_node = marginalize(product_node, [0], prune=prune)

    # Scope query should now only be [1]
    assert marginalized_product_node.scope.query == [1]

    if prune:
        # If pruning, Gaussian should be returned
        assert isinstance(marginalized_product_node, Normal)
    else:
        # Else ProductNode should be returned, with single child
        assert isinstance(marginalized_product_node, ProductNode)
        assert len(marginalized_product_node.inputs) == 1


if __name__ == "__main__":
    unittest.main()
