import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext, to_one_hot


def test_differentiable_sampling_path_produces_finite_probabilities():
    leaf = Bernoulli(scope=Scope([0]), out_channels=3, num_repetitions=2)
    n_samples = 9
    data = torch.full((n_samples, 1), float("nan"))
    sampling_ctx = SamplingContext(
        channel_index=to_one_hot(
            torch.randint(low=0, high=leaf.out_shape.channels, size=(n_samples, 1)),
            dim=-1,
            dim_size=leaf.out_shape.channels,
        ),
        mask=torch.ones((n_samples, 1), dtype=torch.bool),
        repetition_index=to_one_hot(
            torch.randint(low=0, high=leaf.out_shape.repetitions, size=(n_samples,)),
            dim=-1,
            dim_size=leaf.out_shape.repetitions,
        ),
        is_differentiable=True,
        is_mpe=False,
    )

    samples = leaf._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())

    assert samples.shape == (n_samples, 1)
    assert torch.isfinite(samples).all()
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "data, weights",
    [
        # EM case: a 0-observation with negligible responsibility rounds p_est to exactly 1.0 in float32
        ([[1.0], [1.0], [0.0]], [1.0, 1.0, 1e-12]),
        # all observations are 1
        ([[1.0], [1.0], [1.0]], [1.0, 1.0, 1.0]),
    ],
)
def test_mle_keeps_probability_strictly_below_one(data, weights, dtype):
    """An estimate of exactly 1 used to be stored as logits = +inf, so x = 0 had a non-finite likelihood."""
    leaf = Bernoulli(scope=Scope([0]), out_channels=1, num_repetitions=1).to(dtype)
    leaf.maximum_likelihood_estimation(
        torch.tensor(data, dtype=dtype), weights=torch.tensor(weights, dtype=dtype).view(-1, 1, 1, 1)
    )

    assert torch.isfinite(leaf.logits).all()
    assert torch.isfinite(leaf.log_likelihood(torch.tensor([[0.0]], dtype=dtype))).all()
