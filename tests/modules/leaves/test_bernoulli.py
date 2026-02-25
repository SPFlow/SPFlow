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
