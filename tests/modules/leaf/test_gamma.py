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
from spflow.modules.layer.leaf.gamma import Gamma as GammaLayer
from spflow.modules.node.leaf.gamma import Gamma as GammaNode

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5


def make_params(module_type: str, alpha=None, beta=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create parameters for a gamma distribution.

    If alpha and beta are not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        alpha: Mean of the distribution.
        beta: Standard deviation of the distribution.
    """
    if module_type == "node":
        if alpha is not None and beta is not None:
            assert alpha.shape == (1,)
            assert beta.shape == (1,)
            return alpha, beta
        else:
            return torch.rand(1) + 1e-8, torch.rand(1) + 1e-8
    elif module_type == "layer":
        if alpha is not None and beta is not None:
            assert alpha.shape == (NUM_SCOPES, NUM_LEAVES)
            assert beta.shape == (NUM_SCOPES, NUM_LEAVES)
            return alpha, beta
        else:
            return torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8, torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, alpha=None, beta=None) -> Union[GammaNode, GammaLayer]:
    """
    Create a Gamma leaf node.

    Args:
        module_type: Type of the module.
        alpha: Mean of the distribution.
        beta: Standard deviation of the distribution.
    """
    if module_type == "node":
        alpha = alpha if alpha is not None else torch.rand(1) + 1e-8
        beta = beta if beta is not None else torch.rand(1) + 1e-8
        scope = Scope([1])
        return GammaNode(scope=scope, alpha=alpha, beta=beta)
    elif module_type == "layer":
        alpha = alpha if alpha is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        beta = beta if beta is not None else torch.rand(NUM_SCOPES, NUM_LEAVES) + 1e-8
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return GammaLayer(scope=scope, alpha=alpha, beta=beta)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(alpha=None, beta=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a gamma distribution.

    Args:
        module_type: Type of the module.
        alpha: Mean of the distribution.
        beta: Standard deviation of the distribution.
        n_samples: Number of samples to generate.
    """
    alpha = alpha if alpha is not None else torch.rand(TOTAL_SCOPES) + 1e-8
    beta = beta if beta is not None else torch.rand(TOTAL_SCOPES) + 1e-8

    return torch.distributions.Gamma(concentration=alpha, rate=beta).sample((n_samples,))


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a gamma distribution."""
    evaluate_log_likelihood(make_leaf(module_type), make_data())


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a gamma distribution."""
    alpha, beta = make_params(module_type)
    leaf = make_leaf(module_type, alpha=alpha, beta=beta)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        evaluate_samples(leaf, data, is_mpe=is_mpe, sampling_ctx=sampling_ctx)


@pytest.mark.parametrize("bias_correction, module_type", product([False, True], ["node", "layer"]))
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a gamma distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    alpha = torch.rand(TOTAL_SCOPES)
    beta = torch.rand(TOTAL_SCOPES)
    data = make_data(alpha, beta, n_samples=1000)
    maximum_likelihood_estimation(leaf, data, bias_correction=bias_correction)
    assert torch.isclose(leaf.distribution.alpha, alpha[leaf.scope.query].unsqueeze(1), atol=1e-2).all()
    assert torch.isclose(leaf.distribution.beta, beta[leaf.scope.query].unsqueeze(1), atol=1e-2).all()


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a gamma distribution."""
    # Check that parameters are set correctly
    alpha, beta = make_params(module_type)
    leaf = make_leaf(module_type=module_type, alpha=alpha, beta=beta)
    assert torch.isclose(leaf.distribution.alpha, alpha).all()
    assert torch.isclose(leaf.distribution.beta, beta).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, alpha=alpha, beta=-1.0 * beta)  # negative beta
        make_leaf(module_type=module_type, alpha=beta, beta=0.0 * beta)  # zero beta
        make_leaf(module_type=module_type, alpha=torch.full(alpha.shape, torch.nan), beta=beta)  # nan alpha
        make_leaf(module_type=module_type, alpha=alpha, beta=torch.full(beta.shape, torch.nan))  # nan beta
        make_leaf(module_type=module_type, alpha=alpha, beta=beta.unsqueeze(0))  # wrong beta shape
        make_leaf(module_type=module_type, alpha=alpha.unsqueeze(0), beta=beta)  # wrong alpha shape
        make_leaf(module_type=module_type, alpha=None, beta=beta)  # missing alpha
        make_leaf(module_type=module_type, alpha=alpha, beta=None)  # missing beta


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_requires_grad(module_type: str):
    """Test whether the alpha and beta of a gamma distribution require gradients."""
    leaf = make_leaf(module_type)
    assert leaf.distribution.alpha.requires_grad
    assert leaf.distribution.beta.requires_grad


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a gamma distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.alpha.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)
        assert leaf_marg.distribution.beta.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
