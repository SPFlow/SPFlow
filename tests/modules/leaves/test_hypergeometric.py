from itertools import product

import pytest
import torch

from spflow.exceptions import InvalidParameterCombinationError
from spflow.meta import Scope
from spflow.modules.leaves import Hypergeometric

num_repetition_values = [1, 4]
out_channels_values = [1, 5]
out_features_values = [1, 6]


def make_params(
    out_features: int,
    out_channels: int,
    num_repetitions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create parameters for a hypergeometric distribution.

    Args:
        out_features: Number of features.
        out_channels: Number of channels.
        num_repetitions: Number of repetitions.

    Returns:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    shape = (out_features, out_channels, num_repetitions)
    N = torch.randint(10, 100, shape)
    K = torch.randint(1, int(N.max().item()), shape).clamp(max=N - 1)
    n = torch.randint(1, int(N.max().item()), shape).clamp(max=N - 1)
    return K, N, n


def make_leaf(K, N, n) -> Hypergeometric:
    """Create a Hypergeometric leaves node.

    Args:
        K: Number of success states in the population.
        N: Population size.
        n: Number of draws.
    """
    scope = Scope(list(range(K.shape[0])))
    return Hypergeometric(scope=scope, K=K, N=N, n=n)


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=-1.0 * N, n=n).distribution()


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_n(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative n."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=-1.0 * n).distribution()


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_negative_K(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with negative K."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=-1.0 * K, N=N, n=n).distribution()


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_n_greater_than_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with n > N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=N + 1).distribution()


@pytest.mark.parametrize(
    "out_features,out_channels,num_repetitions",
    product(out_features_values, out_channels_values, num_repetition_values),
)
def test_constructor_K_greater_than_N(out_features: int, out_channels: int, num_repetitions: int):
    """Test the constructor of a Hypergeometric distribution with K > N."""
    K, N, n = make_params(out_features, out_channels, num_repetitions)
    with pytest.raises(ValueError):
        make_leaf(K=N + 1, N=N, n=n).distribution()


def test_hypergeometric_missing_parameters():
    """Test that Hypergeometric raises InvalidParameterCombinationError when parameters are missing."""
    scope = Scope([0])
    K = torch.tensor([[5.0]])
    N = torch.tensor([[10.0]])
    with pytest.raises(InvalidParameterCombinationError):
        Hypergeometric(scope=scope, K=K, N=N, n=None)


def test_hypergeometric_log_likelihood_handles_nans_without_broadcast_errors():
    """Hypergeometric log_likelihood should marginalize NaNs without shape/broadcast issues."""
    out_features, out_channels, num_repetitions = 3, 2, 4
    shape = (out_features, out_channels, num_repetitions)
    N = torch.full(shape, 20.0)
    K = torch.full(shape, 7.0)
    n = torch.full(shape, 5.0)
    leaf = make_leaf(K=K, N=N, n=n)

    data = torch.zeros(6, out_features)
    data[2, 1] = torch.nan
    data[4, 0] = torch.nan
    original = data.clone()

    log_prob = leaf.log_likelihood(data)

    assert log_prob.shape == (data.shape[0], out_features, out_channels, num_repetitions)
    torch.testing.assert_close(data, original, equal_nan=True)

    marg_mask = torch.isnan(data).unsqueeze(2).unsqueeze(-1).expand_as(log_prob)
    assert marg_mask.any()
    torch.testing.assert_close(log_prob[marg_mask], torch.zeros_like(log_prob[marg_mask]))


def test_align_support_mask_expands_lower_rank_mask():
    """Support masks with fewer dims are reshaped/sliced/expanded to match data."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    distribution = make_leaf(K=K, N=N, n=n).distribution()

    mask = torch.tensor([[True, False]])
    data = torch.zeros((2, 2, 3))
    aligned = distribution._align_support_mask(mask, data)

    assert aligned.shape == data.shape
    assert torch.all(aligned[:, 0, :])
    assert torch.all(~aligned[:, 1, :])


def test_align_support_mask_raises_on_rank_mismatch():
    """Support mask rank greater than data rank should fail early."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    distribution = make_leaf(K=K, N=N, n=n).distribution()

    mask = torch.ones((2, 2, 2), dtype=torch.bool)
    data = torch.zeros((2, 2))
    with pytest.raises(RuntimeError):
        distribution._align_support_mask(mask, data)


def test_align_support_mask_raises_on_incompatible_shapes():
    """Same-rank but non-broadcastable masks should raise a clear error."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    distribution = make_leaf(K=K, N=N, n=n).distribution()

    mask = torch.ones((2, 3), dtype=torch.bool)
    data = torch.zeros((2, 4))
    with pytest.raises(RuntimeError):
        distribution._align_support_mask(mask, data)


def test_hypergeometric_check_support_raises_for_out_of_support_values():
    """check_support should reject non-integer/out-of-range values."""
    shape = (2, 1, 1)
    K = torch.full(shape, 4.0)
    N = torch.full(shape, 10.0)
    n = torch.full(shape, 3.0)
    leaf = make_leaf(K=K, N=N, n=n)

    data = torch.tensor([[0.0, 1.0], [3.0, 5.0]])
    with pytest.raises(ValueError):
        leaf.distribution().check_support(data)


def test_hypergeometric_check_support_delegates_and_returns_mask():
    """Leaf check_support delegates and returns a boolean validity mask."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    leaf = make_leaf(K=K, N=N, n=n)

    data = torch.tensor([[0.0], [1.0], [2.0], [float("nan")]])
    mask = leaf.check_support(data)
    assert mask.dtype == torch.bool
    assert torch.all(mask)


def test_hypergeometric_sample_accepts_int_sample_count():
    """Sampling should also accept integer sample counts (not only tuples)."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    distribution = make_leaf(K=K, N=N, n=n).distribution()

    samples = distribution.sample(7)
    assert samples.shape == (7, 1, 1, 1)


def test_constructor_rejects_non_integer_N():
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.5)
    n = torch.full((1, 1, 1), 2.0)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_constructor_rejects_non_integer_K():
    K = torch.full((1, 1, 1), 2.5)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_constructor_rejects_non_integer_n():
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.5)
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_constructor_rejects_scopewise_inconsistent_N():
    K = torch.tensor([[[2.0]], [[2.0]]])
    N = torch.tensor([[[5.0]], [[6.0]]])
    n = torch.tensor([[[2.0]], [[2.0]]])
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_constructor_rejects_scopewise_inconsistent_K():
    K = torch.tensor([[[2.0]], [[3.0]]])
    N = torch.tensor([[[5.0]], [[5.0]]])
    n = torch.tensor([[[2.0]], [[2.0]]])
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_constructor_rejects_scopewise_inconsistent_n():
    K = torch.tensor([[[2.0]], [[2.0]]])
    N = torch.tensor([[[5.0]], [[5.0]]])
    n = torch.tensor([[[2.0]], [[3.0]]])
    with pytest.raises(ValueError):
        make_leaf(K=K, N=N, n=n)


def test_set_mle_parameters_is_noop_for_fixed_hypergeometric_params():
    """Hypergeometric parameters are fixed buffers; setting MLE params is a no-op."""
    K = torch.full((1, 1, 1), 2.0)
    N = torch.full((1, 1, 1), 5.0)
    n = torch.full((1, 1, 1), 2.0)
    leaf = make_leaf(K=K, N=N, n=n)

    original = leaf.params()
    leaf._set_mle_parameters(
        {"K": torch.full_like(K, 1.0), "N": torch.full_like(N, 8.0), "n": torch.full_like(n, 1.0)}
    )
    after = leaf.params()

    torch.testing.assert_close(after["K"], original["K"])
    torch.testing.assert_close(after["N"], original["N"])
    torch.testing.assert_close(after["n"], original["n"])
