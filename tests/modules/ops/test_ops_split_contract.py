import pytest
import torch

from spflow.modules.ops import SplitConsecutive, SplitInterleaved
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.modules.module_contract_data import SPLIT_PARAMS
from tests.modules.test_helpers.builders import build_split
from tests.utils.leaves import make_normal_data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("out_channels,out_features,num_splits,num_reps", SPLIT_PARAMS)
@pytest.mark.parametrize("split_type", [SplitConsecutive, SplitInterleaved])
def test_split_loglikelihood_contract(
    out_channels: int, out_features: int, num_splits: int, num_reps: int, split_type
):
    if num_splits > out_features:
        # Builder-level contract excludes degenerate configs with empty partitions.
        pytest.skip("Invalid split config for contract checks: num_splits > out_features")

    module = build_split(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        num_splits=num_splits,
        split_type=split_type,
    )
    data = make_normal_data(out_features=out_features)
    lls = module.log_likelihood(data)
    assert isinstance(lls, list)
    assert len(lls) > 0
    # All split outputs together must form a lossless partition of feature axis.
    assert sum(part.shape[1] for part in lls) == out_features
    for part in lls:
        assert part.shape[0] == data.shape[0]
        assert torch.isfinite(part).all()


@pytest.mark.parametrize("out_channels,out_features,num_splits,num_reps", SPLIT_PARAMS)
@pytest.mark.parametrize("split_type", [SplitConsecutive, SplitInterleaved])
def test_split_sampling_contract(
    out_channels: int, out_features: int, num_splits: int, num_reps: int, split_type
):
    if num_splits > out_features:
        # Keep parameterized matrix focused on valid routing behavior.
        pytest.skip("Invalid split config for contract checks: num_splits > out_features")

    module = build_split(
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=num_reps,
        num_splits=num_splits,
        split_type=split_type,
    )
    n_samples = 8
    sampling_ctx = SamplingContext(
        channel_index=torch.randint(0, module.out_shape.channels, (n_samples, module.out_shape.features)),
        mask=torch.ones((n_samples, module.out_shape.features), dtype=torch.bool),
        repetition_index=torch.randint(0, num_reps, (n_samples,)),
    )
    samples = module._sample(
        data=torch.full((n_samples, out_features), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
    )
    # Sampling API must always reconstruct full-width samples regardless of split type.
    assert samples.shape == (n_samples, out_features)
    assert torch.isfinite(samples).all()
