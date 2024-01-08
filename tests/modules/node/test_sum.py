import unittest

import pytest
from spflow.modules.node.leaf import Gaussian
from spflow.meta.data import Scope
from spflow import tensor as T
from spflow import log_likelihood, sample, marginalize
from spflow.learn import expectation_maximization
from spflow.modules.node import SumNode, ProductNode
from tests.fixtures import backend


def make_sum(num_inputs=2, weights=None, children=None, scope=0):
    if weights is None:
        weights = [1 / num_inputs] * num_inputs
    if children is None:
        children = [make_leaf(scope=scope) for _ in range(num_inputs)]
    return SumNode(children=children, weights=weights)


def make_leaf(mean=0.0, std=1.0, scope=0):
    return Gaussian(scope=Scope([scope]), mean=mean, std=std)


def make_data(mean=0.0, std=1.0, num_samples=10, dim=2):
    return T.randn(num_samples, dim) * std + mean


def test_log_likelihood(backend):
    sum_node = make_sum()
    data = make_data(dim=1)
    lls = log_likelihood(sum_node, data)
    assert lls.shape == (data.shape[0], 1)


def test_sample(backend):
    sum_node = make_sum()
    samples = sample(sum_node, num_samples=100)
    assert samples.shape == (100, 1)


def test_expectation_maximization():
    with T.backend_context(T.Backend.PYTORCH):
        sum_node = make_sum()
        data = make_data(dim=1)
        expectation_maximization(sum_node, data, max_steps=10)


def test_constructor(backend):
    # Check invalid parameters
    with pytest.raises(ValueError):
        sum_node = make_sum(weights=[-0.5, 0.5, 1.0])

    with pytest.raises(ValueError):
        sum_node = make_sum(weights=[-0.5, 1.5])

    with pytest.raises(ValueError):
        sum_node = make_sum(weights=[0.5, 0.2])

    with pytest.raises(ValueError):
        sum_node = make_sum(weights=[0.2, 0.5])

    with pytest.raises(ValueError):
        sum_node = make_sum(children=[make_leaf(scope=0), make_leaf(scope=1)])


@pytest.mark.parametrize("prune", [True, False])
def test_marginalize(backend, prune):
    # Sum over two products over two gaussians
    sum_node = SumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(scope=Scope([0]), mean=0.0, std=1.0),
                    Gaussian(scope=Scope([1]), mean=0.0, std=1.0),
                ]
            ),
            ProductNode(
                children=[
                    Gaussian(scope=Scope([0]), mean=0.0, std=1.0),
                    Gaussian(scope=Scope([1]), mean=0.0, std=1.0),
                ]
            ),
        ]
    )

    # Marginalize first scope
    marginalized_sum_node = marginalize(sum_node, [0], prune=prune)

    # Scope query should now only be [1]
    assert marginalized_sum_node.scope.query == [1]

    if prune:
        # If pruning, Gaussian should be returned
        assert isinstance(marginalized_sum_node, SumNode)
        assert len(marginalized_sum_node.children) == 2
        assert isinstance(marginalized_sum_node.children[0], Gaussian)
        assert isinstance(marginalized_sum_node.children[1], Gaussian)
    else:
        # Else ProductNode should be returned, with single child
        assert isinstance(marginalized_sum_node, SumNode)
        assert isinstance(marginalized_sum_node.children[0], ProductNode)
        assert isinstance(marginalized_sum_node.children[1], ProductNode)
        assert isinstance(marginalized_sum_node.children[0].children[0], Gaussian)
        assert isinstance(marginalized_sum_node.children[1].children[0], Gaussian)
        assert len(marginalized_sum_node.children[0].children) == 1
        assert len(marginalized_sum_node.children[1].children) == 1


if __name__ == "__main__":
    unittest.main()
