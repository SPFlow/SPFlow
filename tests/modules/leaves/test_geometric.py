"""Regression tests for Geometric leaf MLE edge cases."""

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves.geometric import Geometric


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias_correction", [True, False])
@pytest.mark.parametrize(
    "data, weights",
    [
        # EM case: a k > 0 observation with negligible responsibility rounds p_est to exactly 1.0 in float32
        ([[0.0], [0.0], [3.0]], [1.0, 1.0, 1e-12]),
        # all observations are 0
        ([[0.0], [0.0], [0.0]], [1.0, 1.0, 1.0]),
    ],
)
def test_mle_keeps_probability_strictly_below_one(data, weights, bias_correction, dtype):
    """An estimate of exactly 1 used to make the log-likelihood of every k > 0 equal to -inf."""
    leaf = Geometric(scope=Scope([0]), out_channels=1, num_repetitions=1).to(dtype)
    leaf.maximum_likelihood_estimation(
        torch.tensor(data, dtype=dtype),
        weights=torch.tensor(weights, dtype=dtype).view(-1, 1, 1, 1),
        bias_correction=bias_correction,
    )

    assert torch.isfinite(leaf.logits).all()
    # finite logits alone are not enough: torch's Geometric.log_prob uses sigmoid(logits)
    assert torch.isfinite(leaf.log_likelihood(torch.tensor([[3.0]], dtype=dtype))).all()
