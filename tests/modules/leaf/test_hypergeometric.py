import unittest
from itertools import product

from typing import Union
from spflow.meta.dispatch import init_default_sampling_context
from tests.modules.node.leaf.utils import evaluate_log_likelihood
from tests.fixtures import set_seed
import pytest
import torch
from pytest import raises
from scipy.stats import hypergeom

from spflow import maximum_likelihood_estimation, sample, marginalize, log_likelihood
from spflow.meta.data import Scope
from spflow.modules.layer.leaf.hypergeometric import Hypergeometric as HypergeometricLayer
from spflow.modules.node.leaf.hypergeometric import Hypergeometric as HypergeometricNode

# Constants
NUM_LEAVES = 2
NUM_SCOPES = 3
TOTAL_SCOPES = 5
K_TENSOR = torch.tensor([4.0]).reshape(1,)
N_TENSOR = torch.tensor([6.0]).reshape(1,)
n_TENSOR = torch.tensor([5.0]).reshape(1,)


def make_params(module_type: str, K=None, N=None, n=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create parameters for a hypergeometric distribution.

    If K, N and n are not provided, they are randomly initialized, according to the module type.

    Args:
        module_type: Type of the module, can be "node" or "layer".
        K: PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
        N: PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
        n: PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
    """
    if module_type == "node":
        if K is not None and N is not None and n is not None:
            assert K.shape == (1,)
            assert N.shape == (1,)
            assert n.shape == (1,)
            return K, N, n
        else:
            return K_TENSOR, N_TENSOR, n_TENSOR
    elif module_type == "layer":
        if K is not None and N is not None and n is not None:
            assert K.shape == (NUM_SCOPES, NUM_LEAVES)
            assert N.shape == (NUM_SCOPES, NUM_LEAVES)
            assert n.shape == (NUM_SCOPES, NUM_LEAVES)
            return K, N, n
        else:
            return K_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES), N_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES), n_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_leaf(module_type: str, K=None, N=None, n=None) -> Union[HypergeometricNode, HypergeometricLayer]:
    """
    Create a Hypergeometric leaf node.

    Args:
        module_type: Type of the module.
        K: PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
        N: PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
        n: PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
    """
    if module_type == "node":
        K = K if K is not None else K_TENSOR
        N = N if N is not None else N_TENSOR
        n = n if n is not None else n_TENSOR
        scope = Scope([1])
        return HypergeometricNode(scope=scope, K=K, N=N, n=n)
    elif module_type == "layer":
        K = K if K is not None else K_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES)
        N = N if N is not None else N_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES)
        n = n if n is not None else n_TENSOR.repeat(NUM_SCOPES, NUM_LEAVES)
        scope = Scope(list(range(1, NUM_SCOPES + 1)))
        return HypergeometricLayer(scope=scope, K=K, N=N, n=n)
    else:
        raise ValueError(f"Invalid module_type: {module_type}")


def make_data(K=None, N=None, n=None, n_samples=5) -> torch.Tensor:
    """
    Generate data from a hypergeometric distribution.

    Args:
        module_type: Type of the module.
        K: PyTorch tensor specifying the total numbers of entities (in the populations), greater or equal to 0.
        N: PyTorch tensor specifying the numbers of entities with property of interest (in the populations), greater or equal to zero and less than or equal to N.
        n: PyTorch tensor specifying the numbers of draws, greater of equal to zero and less than or equal to N.
        n_samples: Number of samples to generate.
    """
    K = K if K is not None else K_TENSOR.repeat(TOTAL_SCOPES)
    N = N if N is not None else N_TENSOR.repeat(TOTAL_SCOPES)
    n = n if n is not None else n_TENSOR.repeat(TOTAL_SCOPES)

    # scipy: M = our N, scipy N = our n, scipy n = our K

    return hypergeom(M=N.type(torch.int32), N=n.type(torch.int32), n=K.type(torch.int32)).rvs(size=(n_samples, TOTAL_SCOPES))


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_log_likelihood(module_type: str):
    """Test the log likelihood of a hypergeometric distribution."""
    node = make_leaf(module_type)
    data = torch.tensor(make_data())

    lls = log_likelihood(node, data, check_support=True)
    assert lls.shape == (data.shape[0], len(node.scope.query), node.event_shape[1])
    assert torch.isfinite(lls).all()

    # compare with scipy
    scipy_lls = hypergeom(M=node.distribution.N, N=node.distribution.n, n=node.distribution.K).logpmf(data[:,node.scope.query].unsqueeze(2))
    assert torch.allclose(lls, torch.tensor(scipy_lls).to(lls.dtype), atol=1e-4)


@pytest.mark.parametrize("module_type,is_mpe", product(["node", "layer"], [True, False]))
def test_sample(module_type: str, is_mpe: bool):
    """Test sampling from a hypergeometric distribution."""
    K, N, n = make_params(module_type)
    leaf = make_leaf(module_type, K, N, n)

    n_samples = 5000
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, n=n_samples)

    leaf_mean = n*K/N
    leaf_std = torch.sqrt((n*(N-n)*(N-K))/(N**2*(N-1)))

    if module_type == "node":
        # Kake space for num_leaves dimension in node test case
        leaf_mean.unsqueeze_(-1)
        leaf_std.unsqueeze_(-1)

    for i in range(NUM_LEAVES):
        data = torch.full((n_samples, TOTAL_SCOPES), torch.nan)
        sampling_ctx.output_ids = torch.full((n_samples, NUM_SCOPES), fill_value=i)
        samples = sample(leaf, data, is_mpe=is_mpe, check_support=True, sampling_ctx=sampling_ctx)

        if is_mpe:
            # MPE doesn't use sample method so we don't check the output
            pass
        else:
            assert torch.isclose(samples[:, leaf.scope.query].mean(0), leaf_mean[:, i], atol=1e-1).all()
            assert torch.isclose(samples[:, leaf.scope.query].std(0), leaf_std[:, i], atol=3e-1).all()

        if module_type == "node":
            # Break after first round since nodes only have a single leaf per scope
            break


@pytest.mark.parametrize("bias_correction, module_type", product([True, False], ["node", "layer"]))
def test_maximum_likelihood_estimation(bias_correction: bool, module_type: str):
    """Test maximum likelihood estimation of a hypergeometric distribution.

    Args:
        bias_correction: Whether to use bias correction in the estimation.
    """
    leaf = make_leaf(module_type)
    data = make_data(n_samples=1000)
    maximum_likelihood_estimation(leaf, torch.tensor(data), bias_correction=bias_correction)



@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_constructor(module_type: str):
    """Test the constructor of a hypergeometric distribution."""
    # Check that parameters are set correctly
    K,N,n = make_params(module_type)
    leaf = make_leaf(module_type=module_type, K=K, N=N, n=n)
    assert torch.isclose(leaf.distribution.K, K).all()
    assert torch.isclose(leaf.distribution.N, N).all()
    assert torch.isclose(leaf.distribution.n, n).all()

    # Check invalid parameters
    with raises(ValueError):
        make_leaf(module_type=module_type, K=K, N=-1.0 * N, n=n)  # negative N
        make_leaf(module_type=module_type, K=K, N=N, n=-1.0 * n)  # negative n
        make_leaf(module_type=module_type, K=-1.0 * K, N=N, n=n)  # negative K
        make_leaf(module_type=module_type, K=K, N=N, n=N+1)  # n > N
        make_leaf(module_type=module_type, K=N+1, N=N, n=n)  # K > N
        make_leaf(module_type=module_type, K=torch.full(K.shape, torch.nan), N=N, n=n)  # nan K
        make_leaf(module_type=module_type, K=K, N=torch.full(N.shape, torch.nan), n=n)  # nan N
        make_leaf(module_type=module_type, K=K, N=N, n=torch.full(n.shape, torch.nan))  # nan n
        make_leaf(module_type=module_type, K=K.unsqueeze(0), N=N, n=n)  # wrong K shape
        make_leaf(module_type=module_type, K=K, N=N.unsqueeze(0), n=n)  # wrong N shape
        make_leaf(module_type=module_type, K=K, N=N, n=n.unsqueeze(0))  # wrong n shape


@pytest.mark.parametrize("module_type", ["node", "layer"])
def test_marginalize(module_type: str):
    """Test marginalization of a hypergeometric distribution."""
    leaf = make_leaf(module_type)
    scope_og = leaf.scope.copy()
    marg_rvs = [1, 2]
    leaf_marg = marginalize(leaf, marg_rvs)

    if module_type == "node":
        assert leaf_marg == None
    else:
        assert leaf_marg.distribution.K.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)
        assert leaf_marg.distribution.N.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)
        assert leaf_marg.distribution.n.shape == (NUM_SCOPES - len(marg_rvs), NUM_LEAVES)

        # TODO: ensure, that the correct scopes were marginalized
        assert leaf_marg.scope.query == [q for q in scope_og.query if q not in marg_rvs]


if __name__ == "__main__":
    unittest.main()
