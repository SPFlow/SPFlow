import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.modules.node.leaf.utils import evaluate_log_likelihood, evaluate_samples
from tests.fixtures import set_seed
import pytest
import torch
from pytest import raises

from spflow import maximum_likelihood_estimation, sample, marginalize
from spflow.meta.data import Scope
from spflow.modules.layer.leaf.geometric import Geometric as GeometricLayer
from spflow.modules.node.leaf.geometric_new import Geometric as GeometricNode

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5


def make_params(module_type: str, p=None) -> torch.Tensor:
    """
    Create parameters for a geometric distribution.

    If p is not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        p: success probabilities of the distribution.
    """
    if module_type == "node":
        if p is not None:
            assert p.shape == (1,)
            return p
        else:
            return torch.rand(1) + 1e-8
    elif module_type == "layer":
        if  p is not None:
            assert p.shape == (NUM_SCOPES, NUM_LEAVES)
            return p
        else:
            return torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, p=None) -> Union[GeometricNode, GeometricLayer]:
    """
    Create a Geometric leaf node.

    Args:
        module_type: Type of the module.
        p: Success probabilities of the distribution.
    """
    if module_type == "node":
        p = p if p is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return GeometricNode(scope=scope, p=p)
    elif module_type == "layer":
        p = p if p is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return GeometricLayer(scope=scope, p=p)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(p=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a geometric distribution.

    Args:
        module_type: Type of the module.
        p: Success probabilities of the distribution.
        n_samples: Number of samples to generate.
    """
    p = p if p is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.Geometric(probs=p).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a geometric distribution."""
    evaluate_log_likelihood(make_leaf(module_type), make_data())


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a bernoulli distribution."""
    p = make_params(module_type)
    leaf = make_leaf(module_type, p=p)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "layer"]))
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a geometric distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    p = torch.rand(TOTAL_SCOPES)
    data = make_data(p, n_samples=5000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.distribution.p, p[leaf.scope.query].unsqueeze(1), atol=1e-2).all()


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a geometric distribution."""
    # Check that parameters are set correctly
    p = make_params(module_type)
    leaf = make_leaf(module_type=module_type, p=p)
    assert torch.isclose(leaf.distribution.p, p).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, p=-1.0 * p)  # negative p
        make_leaf(module_type=module_type, p=1.0 + p)  # p gt 1.0
        make_leaf(module_type=module_type, p=torch.full(p.shape, torch.nan))  # nan p
        make_leaf(module_type=module_type, p=p.unsqueeze(0))  # wrong p shape
        make_leaf(module_type=module_type, p=None)  # missing p


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_requires_grad(module_type: str):
    """Test whether the mean and p of a geometric distribution require gradients."""
    leaf = make_leaf(module_type)
    assert leaf.distribution.p.requires_grad


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a geometric distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.p.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
