import itertools

import torch

from spflow.meta import Scope
from spflow.modules.leaves import CLTree


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
