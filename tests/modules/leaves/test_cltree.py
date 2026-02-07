import itertools

import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves import CLTree
from spflow.modules.leaves.cltree import (
    _compute_orders_from_parents,
    _prim_maximum_spanning_tree,
    _validate_discrete_values,
)
from spflow.utils.sampling_context import SamplingContext


def _make_chain_cltree(*, K: int = 3) -> CLTree:
    # Scope order matches CLTree internal feature order.
    scope = Scope([0, 1, 2])
    parents = torch.tensor([-1, 0, 1], dtype=torch.long)

    # Define a simple, non-degenerate distribution.
    p_root = torch.tensor([0.2, 0.5, 0.3])
    p_1_given_0 = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )  # rows: x1, cols: x0
    p_2_given_1 = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )  # rows: x2, cols: x1

    log_cpt = torch.empty((3, 1, 1, K, K), dtype=torch.float64)
    log_cpt[0, 0, 0, :, :] = p_root.log().unsqueeze(-1).expand(K, K)
    log_cpt[1, 0, 0, :, :] = p_1_given_0.log()
    log_cpt[2, 0, 0, :, :] = p_2_given_1.log()

    return CLTree(scope=scope, out_channels=1, num_repetitions=1, K=K, parents=parents, log_cpt=log_cpt)


def _joint_log_prob_chain(log_cpt: torch.Tensor, x: tuple[int, int, int]) -> torch.Tensor:
    x0, x1, x2 = x
    return log_cpt[0, 0, 0, x0, 0] + log_cpt[1, 0, 0, x1, x0] + log_cpt[2, 0, 0, x2, x1]


def _bruteforce_log_prob_with_nans(log_cpt: torch.Tensor, evidence: torch.Tensor, K: int) -> torch.Tensor:
    # evidence shape: (3,), float with NaNs
    choices: list[list[int]] = []
    for v in evidence.tolist():
        if v != v:  # NaN
            choices.append(list(range(K)))
        else:
            choices.append([int(v)])

    vals = []
    for x0, x1, x2 in itertools.product(*choices):
        vals.append(_joint_log_prob_chain(log_cpt, (x0, x1, x2)))
    return torch.logsumexp(torch.stack(vals), dim=0)


def test_log_likelihood_matches_bruteforce_on_tiny_scope():
    K = 3
    node = _make_chain_cltree(K=K)

    # Build a batch with both complete and partially missing rows.
    data = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, float("nan"), 0.0],
            [float("nan"), 2.0, float("nan")],
        ],
        dtype=torch.float64,
    )

    ll = node.log_likelihood(data).sum(dim=1).squeeze(-1).squeeze(-1)  # (N,)
    expected = torch.stack(
        [
            _bruteforce_log_prob_with_nans(node.log_cpt.detach(), data[0], K),
            _bruteforce_log_prob_with_nans(node.log_cpt.detach(), data[1], K),
            _bruteforce_log_prob_with_nans(node.log_cpt.detach(), data[2], K),
        ]
    )
    torch.testing.assert_close(ll, expected, rtol=1e-6, atol=1e-6)


def test_mpe_fills_missing_values_consistently():
    K = 3
    node = _make_chain_cltree(K=K)

    evidence = torch.tensor([[float("nan"), 2.0, float("nan")]], dtype=torch.float64)
    mpe = node.sample(data=evidence.clone(), is_mpe=True)
    mpe_vals = tuple(int(v) for v in mpe[0].tolist())

    # Brute force MAP under evidence x1=2
    best_x = None
    best_lp = None
    for x0 in range(K):
        for x2 in range(K):
            x = (x0, 2, x2)
            lp = _joint_log_prob_chain(node.log_cpt.detach(), x)
            if best_lp is None or lp > best_lp:
                best_lp = lp
                best_x = x

    assert best_x is not None
    assert mpe_vals == best_x


def test_sampling_produces_valid_domain_values():
    K = 3
    node = _make_chain_cltree(K=K)

    samples = node.sample(num_samples=200)
    assert samples.shape == (200, 3)
    assert torch.isfinite(samples).all()
    assert ((samples >= 0) & (samples < K)).all()

    # Conditional sampling preserves evidence.
    evidence = torch.tensor([[1.0, float("nan"), 0.0]] * 50, dtype=torch.float64)
    filled = node.sample(data=evidence.clone(), is_mpe=False)
    torch.testing.assert_close(filled[:, 0], torch.ones(50, dtype=torch.float64))
    torch.testing.assert_close(filled[:, 2], torch.zeros(50, dtype=torch.float64))
    assert ((filled[:, 1] >= 0) & (filled[:, 1] < K)).all()


def test_mle_update_improves_likelihood_on_training_data():
    torch.manual_seed(0)
    K = 3

    # Generate synthetic data from a known model.
    true_node = _make_chain_cltree(K=K)
    data = true_node.sample(num_samples=500).to(torch.float64)

    # Fit structure once, then compare random CPT vs MLE CPT.
    model = CLTree(scope=Scope([0, 1, 2]), out_channels=1, num_repetitions=1, K=K)
    model.fit_structure(data)

    # Random valid CPT initialization.
    rand = torch.rand((3, 1, 1, K, K), dtype=torch.float64)
    rand = rand / rand.sum(dim=3, keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        model.log_cpt.copy_(rand.clamp_min(1e-12).log())

    ll_before = model.log_likelihood(data).sum(dim=1).mean()
    model.maximum_likelihood_estimation(data)
    ll_after = model.log_likelihood(data).sum(dim=1).mean()

    assert ll_after >= ll_before - 1e-6


def test_helper_validation_and_tree_utilities_cover_edge_cases():
    _validate_discrete_values(torch.empty((0, 3)), K=3)
    _validate_discrete_values(torch.full((2, 3), float("nan")), K=3)
    with pytest.raises(InvalidParameterError):
        _validate_discrete_values(torch.tensor([[0.1, 1.0]]), K=3)
    with pytest.raises(InvalidParameterError):
        _validate_discrete_values(torch.tensor([[0.0, 3.0]]), K=3)

    assert _compute_orders_from_parents([]).pre_order == ()
    with pytest.raises(InvalidParameterError):
        _compute_orders_from_parents([-1, -1, 0])
    with pytest.raises(InvalidParameterError):
        _compute_orders_from_parents([-1, 3, 0])
    orders = _compute_orders_from_parents([-1, 0, 0, 1])
    assert orders.pre_order == (0, 1, 3, 2)
    assert orders.post_order == (2, 3, 1, 0)

    with pytest.raises(ShapeError):
        _prim_maximum_spanning_tree(torch.ones(2, 3))
    assert _prim_maximum_spanning_tree(torch.empty((0, 0))) == []
    with pytest.raises(InvalidParameterError):
        _prim_maximum_spanning_tree(torch.zeros(3, 3), root=3)
    with pytest.raises(UnsupportedOperationError):
        _prim_maximum_spanning_tree(torch.full((3, 3), float("-inf")))

    weights = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    assert _prim_maximum_spanning_tree(weights, root=0) == [-1, 0, 1]


def test_init_and_public_api_validation_errors():
    with pytest.raises(InvalidParameterError):
        CLTree(scope=Scope([0, 1]), K=1)
    with pytest.raises(InvalidParameterError):
        CLTree(scope=Scope([0, 1]), K=2, alpha=0.0)
    with pytest.raises(InvalidParameterError):
        CLTree(scope=Scope([0]), K=2)

    with pytest.raises(ShapeError):
        CLTree(scope=Scope([0, 1]), K=2, parents=torch.tensor([-1, 0, 1]))

    with pytest.raises(ShapeError):
        CLTree(scope=Scope([0, 1]), K=2, log_cpt=torch.zeros((2, 1, 1, 2, 3)))

    node = CLTree(scope=Scope([0, 1]), K=2)
    assert node._supported_value == 0.0
    assert "log_cpt" in node.params()
    with pytest.raises(UnsupportedOperationError):
        _ = node._torch_distribution_class
    with pytest.raises(UnsupportedOperationError):
        node._compute_parameter_estimates(
            data=torch.zeros((1, 2)),
            weights=torch.zeros((1, 2, 1, 1)),
            bias_correction=True,
        )


def test_structure_and_data_resolution_checks():
    node = CLTree(scope=Scope([0, 1, 2]), K=3)
    assert not node._has_learned_structure()

    node.parents.data = torch.tensor([-1, 0, 1], dtype=torch.long)
    assert node._has_learned_structure()
    node.parents.data = torch.tensor([-1, 1, 1], dtype=torch.long)
    assert not node._has_learned_structure()

    with pytest.raises(ShapeError):
        node._resolve_scoped_data(torch.zeros((2, 3, 1)))

    node._resolve_scope_columns = lambda num_features: [0, 1]  # type: ignore[method-assign]
    with pytest.raises(ShapeError):
        node._resolve_scoped_data(torch.zeros((2, 3)))


def test_fit_structure_mutual_information_and_log_likelihood_errors():
    node = CLTree(scope=Scope([0, 1, 2]), K=3)
    data = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
    )

    with pytest.raises(ShapeError):
        node._compute_mutual_information(data.to(torch.long), sample_weights=torch.ones((4, 1)))
    with pytest.raises(InvalidParameterError):
        node._compute_mutual_information(
            data.to(torch.long), sample_weights=torch.tensor([1.0, 1.0, float("inf"), 1.0])
        )

    node.fit_structure(data, sample_weights=torch.tensor([1.0, 2.0, 1.0, 1.0]))
    assert node._has_learned_structure()
    assert node.parents.shape[0] == 3

    with pytest.raises(InvalidParameterError):
        node.fit_structure(torch.tensor([[0.0, float("nan"), 1.0], [1.0, 1.0, 0.0]]))

    unfit = CLTree(scope=Scope([0, 1, 2]), K=3)
    with pytest.raises(RuntimeError):
        unfit.log_likelihood(data)


def test_mle_input_checks_and_sampling_context_errors():
    node = _make_chain_cltree(K=3)
    data = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])

    with pytest.raises(ShapeError):
        node.maximum_likelihood_estimation(data, weights=torch.ones((2, 3)))
    with pytest.raises(ShapeError):
        node.maximum_likelihood_estimation(data, weights=torch.ones((3, 3, 1, 1)))
    with pytest.raises(InvalidParameterError):
        node.maximum_likelihood_estimation(
            torch.tensor([[0.0, float("nan"), 2.0], [float("nan"), 1.0, 0.0]]),
            nan_strategy="ignore",
        )

    bad_ctx = SamplingContext(
        channel_index=torch.tensor([[0, 1, 0]], dtype=torch.long),
        mask=torch.tensor([[True, True, True]], dtype=torch.bool),
    )
    with pytest.raises(InvalidParameterError):
        node.sample(data=torch.tensor([[float("nan"), float("nan"), float("nan")]]), sampling_ctx=bad_ctx)

    base = _make_chain_cltree(K=3)
    multi_rep = CLTree(
        scope=Scope([0, 1, 2]),
        out_channels=1,
        num_repetitions=2,
        K=3,
        parents=base.parents.clone(),
        log_cpt=base.log_cpt.detach().repeat(1, 1, 2, 1, 1),
    )
    with pytest.raises(InvalidParameterError):
        multi_rep.sample(data=torch.tensor([[float("nan"), float("nan"), float("nan")]]))
