from itertools import product

import numpy as np
import pytest
import torch

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules import leaves
from spflow.modules.leaves import Normal, Bernoulli
from spflow.modules.ops import SplitMode
from spflow.utils.cache import Cache
from spflow.zoo.rat import RatSPN
from spflow.utils.sampling_context import (
    DifferentiableSamplingContext,
    SamplingContext,
    init_default_sampling_context,
)
from tests.utils.leaves import make_leaf, make_data

depth = [1, 3]
n_region_nodes = [1, 5]
num_leaves = [1, 6]
num_repetitions = [1, 7]
n_root_nodes = [1, 4]
outer_product = [True, False]
split_mode = [None, SplitMode.consecutive(), SplitMode.interleaved()]
leaf_cls_values = [
    # leaves.Bernoulli,
    # leaves.Binomial,
    # leaves.Categorical,
    # leaves.Exponential,
    # leaves.Gamma,
    # leaves.Geometric,
    # leaves.Hypergeometric,
    # leaves.LogNormal,
    # leaves.NegativeBinomial,
    leaves.Normal,
    # leaves.Poisson,
    # leaves.Uniform,
]
params = list(
    product(
        leaf_cls_values,
        depth,
        n_region_nodes,
        num_leaves,
        num_repetitions,
        n_root_nodes,
        outer_product,
        split_mode,
    )
)


def make_rat_spn(
    leaf_cls,
    depth,
    n_region_nodes,
    num_leaves,
    num_repetitions,
    n_root_nodes,
    num_features,
    outer_product,
    split_mode,
):
    depth = depth
    n_region_nodes = n_region_nodes
    num_leaves = num_leaves
    num_repetitions = num_repetitions
    n_root_nodes = n_root_nodes
    num_features = num_features

    leaf_layer = make_leaf(
        cls=leaf_cls, out_channels=num_leaves, out_features=num_features, num_repetitions=num_repetitions
    )

    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,
        num_repetitions=num_repetitions,
        depth=depth,
        outer_product=outer_product,
        split_mode=split_mode,
    )
    return model


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode ", params
)
def test_log_likelihood(leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode):
    num_features = 64
    module = make_rat_spn(
        leaf_cls=leaf_cls,
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,
        outer_product=outer_product,
        split_mode=split_mode,
    )
    assert len(module.scope) == num_features
    data = make_data(cls=leaf_cls, out_features=num_features, n_samples=10)
    # data = data.unsqueeze(1).repeat(1,3,1)
    lls = module.log_likelihood(data)

    # Check that output has expected structure: [batch, features, channels, num_reps]
    assert lls.ndim == 4, f"Expected 4D output, got {lls.ndim}D with shape {lls.shape}"
    assert lls.shape[0] == data.shape[0], f"Batch size mismatch: got {lls.shape[0]}, expected {data.shape[0]}"
    assert (
        lls.shape[1] == module.out_shape.features
    ), f"Out_features mismatch: got {lls.shape[1]}, expected {module.out_shape.features}"
    assert (
        lls.shape[2] == module.out_shape.channels
    ), f"Out_channels mismatch: got {lls.shape[2]}, expected {module.out_shape.channels}"
    assert lls.shape[3] == 1, f"Num_reps mismatch: got {lls.shape[3]}, expected {1}"


@pytest.mark.parametrize(
    "leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode ", params
)
def test_sample(leaf_cls, d, region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode):
    n_samples = 100
    num_features = 64
    module = make_rat_spn(
        leaf_cls=leaf_cls,
        depth=d,
        n_region_nodes=region_nodes,
        num_leaves=leaves,
        num_repetitions=num_reps,
        n_root_nodes=root_nodes,
        num_features=num_features,
        outer_product=outer_product,
        split_mode=split_mode,
    )
    data = torch.full((n_samples, num_features), torch.nan)
    channel_index = torch.randint(
        low=0, high=module.out_shape.channels, size=(n_samples, module.out_shape.features)
    )
    mask = torch.full((n_samples, module.out_shape.features), True)
    repetition_index = torch.randint(low=0, high=num_reps, size=(n_samples,))
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_index=repetition_index)
    samples = module.sample(data=data)
    assert samples.shape == data.shape
    samples_query = samples[:, module.scope.query]
    assert torch.isfinite(samples_query).all()


@pytest.mark.parametrize(
    "region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode ",
    list(product(n_region_nodes, num_leaves, num_repetitions, n_root_nodes, outer_product, split_mode)),
)
def test_multidistribution_input(region_nodes, leaves, num_reps, root_nodes, outer_product, split_mode):
    out_features_1 = 8
    out_features_2 = 10
    depth = 2

    scope_1 = Scope(list(range(0, out_features_1)))
    scope_2 = Scope(list(range(out_features_1, out_features_1 + out_features_2)))

    cls_1 = Normal
    cls_2 = Bernoulli

    module_1 = make_leaf(cls=cls_1, out_channels=leaves, scope=scope_1, num_repetitions=num_reps)
    data_1 = make_data(cls=cls_1, out_features=out_features_1, n_samples=5)

    module_2 = make_leaf(cls=cls_2, out_channels=leaves, scope=scope_2, num_repetitions=num_reps)
    data_2 = make_data(cls=cls_2, out_features=out_features_2, n_samples=5)

    data = torch.cat((data_1, data_2), dim=1)

    model = RatSPN(
        leaf_modules=[module_1, module_2],
        n_root_nodes=root_nodes,
        n_region_nodes=region_nodes,
        num_repetitions=num_reps,
        depth=depth,
        outer_product=outer_product,
        split_mode=split_mode,
    )

    lls = model.log_likelihood(data)

    assert lls.shape == (
        data.shape[0],
        model.out_shape.features,
        model.out_shape.channels,
        1,
    )  # num_reps is 1 after RAT SPN

    repetition_idx = torch.zeros((1,), dtype=torch.long)
    sampling_ctx = init_default_sampling_context(sampling_ctx=None, num_samples=1)
    sampling_ctx.repetition_idx = repetition_idx
    samples = model.sample()

    assert samples.shape == (1, out_features_1 + out_features_2)


def test_rat_spn_feature_to_scope():
    """Test feature_to_scope delegates to root_node."""
    num_features = 64
    leaf_cls = Normal
    leaf_layer = make_leaf(cls=leaf_cls, out_channels=3, out_features=num_features, num_repetitions=1)

    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=2,
        n_region_nodes=4,
        num_repetitions=1,
        depth=1,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )

    # Get feature_to_scope from both RatSPN and root_node
    feature_scopes = model.feature_to_scope
    root_scopes = model.root_node.feature_to_scope

    # Should delegate to root_node's feature_to_scope
    assert np.array_equal(feature_scopes, root_scopes)

    # Check shape matches number of features in scope
    assert feature_scopes.shape == (1, 1)
    assert feature_scopes[0, 0] == model.scope

    # All elements should be Scope objects
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_single_root_node():
    """Test feature_to_scope when n_root_nodes=1."""
    num_features = 4
    leaf_cls = Normal
    leaf_layer = make_leaf(cls=leaf_cls, out_channels=2, out_features=num_features, num_repetitions=1)

    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=1,
        n_region_nodes=3,
        num_repetitions=1,
        depth=1,
        outer_product=False,
        split_mode=SplitMode.interleaved(),
    )

    feature_scopes = model.feature_to_scope

    # Verify delegation to root_node
    assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)

    # Verify shape and Scope objects
    assert len(feature_scopes[0, 0].query) == num_features
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_multiple_repetitions():
    """Test feature_to_scope with num_repetitions > 1."""
    num_features = 64
    leaf_cls = Normal

    for num_reps in [1, 2, 3]:
        leaf_layer = make_leaf(
            cls=leaf_cls, out_channels=4, out_features=num_features, num_repetitions=num_reps
        )

        model = RatSPN(
            leaf_modules=[leaf_layer],
            n_root_nodes=2,
            n_region_nodes=3,
            num_repetitions=num_reps,
            depth=1,
            outer_product=False,
            split_mode=SplitMode.consecutive(),
        )

        feature_scopes = model.feature_to_scope

        # Should delegate correctly regardless of repetitions
        assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
        assert len(feature_scopes[0, 0].query) == num_features
        assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_split_variants():
    """Test feature_to_scope with different split strategies."""
    num_features = 64
    leaf_cls = Normal

    for split_mode_val in [None, SplitMode.consecutive(), SplitMode.interleaved()]:
        leaf_layer = make_leaf(cls=leaf_cls, out_channels=3, out_features=num_features, num_repetitions=1)

        model = RatSPN(
            leaf_modules=[leaf_layer],
            n_root_nodes=2,
            n_region_nodes=4,
            num_repetitions=1,
            depth=1,
            outer_product=False,
            split_mode=split_mode_val,
        )

        feature_scopes = model.feature_to_scope

        # Should work with both split strategies
        assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
        assert len(feature_scopes[0, 0].query) == num_features
        assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


class _DummyOutShape:
    def __init__(self, channels: int):
        self.channels = channels


class _DummyLeaf:
    def __init__(self, channels: int):
        self.out_shape = _DummyOutShape(channels=channels)
        self.scope = Scope([0])


def test_constructor_rejects_invalid_hyperparameters():
    valid_leaf = make_leaf(cls=Normal, out_channels=2, out_features=4, num_repetitions=1)

    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[valid_leaf],
            n_root_nodes=0,
            n_region_nodes=1,
            num_repetitions=1,
            depth=1,
        )

    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[valid_leaf],
            n_root_nodes=1,
            n_region_nodes=0,
            num_repetitions=1,
            depth=1,
        )

    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[_DummyLeaf(channels=0)],
            n_root_nodes=1,
            n_region_nodes=1,
            num_repetitions=1,
            depth=1,
        )

    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[valid_leaf],
            n_root_nodes=1,
            n_region_nodes=1,
            num_repetitions=1,
            depth=1,
            split_mode=SplitMode.consecutive(),
            num_splits=1,
        )


def test_n_out_and_scopes_out_delegate_to_root_node():
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=2,
        num_features=8,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )

    monkey_scopes_out = [module.scope]
    setattr(module.root_node, "scopes_out", monkey_scopes_out)

    assert module.n_out == 1
    assert module.scopes_out == monkey_scopes_out


def test_log_posterior_raises_for_single_class():
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=1,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = make_data(cls=Normal, out_features=6, n_samples=3)

    with pytest.raises(UnsupportedOperationError):
        module.log_posterior(data)


def test_log_posterior_and_predict_proba_shapes():
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=3,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = make_data(cls=Normal, out_features=6, n_samples=5)

    log_post = module.log_posterior(data)
    proba = module.predict_proba(data)

    assert log_post.shape == (data.shape[0], 3)
    assert proba.shape == (data.shape[0], 3)
    assert torch.allclose(proba.sum(dim=1), torch.ones(data.shape[0]), atol=1e-5)


def test_sample_initializes_channel_index_for_mpe_and_stochastic(monkeypatch):
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=3,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = torch.full((4, 6), torch.nan)
    monkeypatch.setattr(
        module.root_node,
        "logits",
        torch.nn.Parameter(torch.zeros(1, module.n_root_nodes, 1)),
    )

    captured: dict[str, SamplingContext] = {}
    original_sample = module.root_node.inputs._sample

    def capture_sample(*, data, sampling_ctx, cache, is_mpe=False):
        captured["ctx"] = sampling_ctx.copy()
        return original_sample(data=data, sampling_ctx=sampling_ctx, cache=cache, is_mpe=is_mpe)

    monkeypatch.setattr(module.root_node.inputs, "_sample", capture_sample)

    module.sample(data=data, is_mpe=True)
    mpe_channel_index = captured["ctx"].channel_index
    assert mpe_channel_index is not None
    assert mpe_channel_index.shape[0] == data.shape[0]

    module.sample(data=data, is_mpe=False)
    random_channel_index = captured["ctx"].channel_index
    assert random_channel_index is not None
    assert random_channel_index.shape[0] == data.shape[0]


def test_sample_raises_when_logits_shape_is_invalid(monkeypatch):
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=3,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = torch.full((2, 6), torch.nan)

    monkeypatch.setattr(
        module.root_node,
        "logits",
        torch.nn.Parameter(torch.zeros(1, module.n_root_nodes + 1, 1)),
    )
    with pytest.raises(InvalidParameterError):
        module.sample(data=data)


def test_rsample_runs_and_backpropagates():
    num_features = 8
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=3,
        num_repetitions=2,
        n_root_nodes=3,
        num_features=num_features,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = torch.full((6, num_features), torch.nan)

    samples = module.rsample(
        data=data,
        diff_method="gumbel",
        hard=False,
    )
    assert torch.isfinite(samples).all()

    loss = samples.square().mean()
    loss.backward()

    assert module.root_node.logits.grad is not None
    assert module.leaf_modules[0].loc.grad is not None
    assert torch.isfinite(module.root_node.logits.grad).all()
    assert torch.isfinite(module.leaf_modules[0].loc.grad).all()


def test_rsample_preserves_evidence_and_tracks_sample_mass():
    num_features = 4
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=3,
        num_repetitions=1,
        n_root_nodes=1,
        num_features=num_features,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = torch.tensor(
        [
            [float("nan"), 1.5, float("nan"), float("nan")],
            [2.0, float("nan"), float("nan"), 4.0],
        ],
        dtype=torch.float32,
    )
    observed_mask = ~torch.isnan(data)
    expected_mass = torch.isnan(data).to(torch.float32)

    sampling_ctx = DifferentiableSamplingContext(num_samples=data.shape[0])
    samples = module._rsample(
        data=data.clone(),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
        is_mpe=False,
    )
    if sampling_ctx.sample_accum is not None and sampling_ctx.sample_mass is not None:
        samples = sampling_ctx.finalize_with_evidence(data.clone())

    assert torch.isfinite(samples).all()
    torch.testing.assert_close(samples[observed_mask], data[observed_mask], rtol=0.0, atol=0.0)
    assert sampling_ctx.sample_mass is not None
    torch.testing.assert_close(
        sampling_ctx.sample_mass,
        expected_mass.to(dtype=sampling_ctx.sample_mass.dtype, device=sampling_ctx.sample_mass.device),
        atol=1e-4,
        rtol=1e-4,
    )


def test_rsample_initializes_root_routing_context(monkeypatch):
    num_features = 6
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=3,
        num_features=num_features,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )

    monkeypatch.setattr(
        module.root_node,
        "logits",
        torch.nn.Parameter(torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)),
    )
    sampling_ctx = DifferentiableSamplingContext(num_samples=3)

    captured: dict[str, torch.Tensor] = {}

    def capture_root_rsample(*, data, sampling_ctx, cache, is_mpe=False):
        del data
        del cache
        del is_mpe
        captured["channel_probs"] = sampling_ctx.channel_probs.clone()
        captured["mask"] = sampling_ctx.mask.clone()
        return torch.empty(0)

    monkeypatch.setattr(module.root_node.inputs, "_rsample", capture_root_rsample)

    module._rsample(
        data=torch.full((3, num_features), torch.nan),
        sampling_ctx=sampling_ctx,
        cache=Cache(),
        is_mpe=True,
    )

    expected = torch.zeros((3, 1, module.n_root_nodes), dtype=captured["channel_probs"].dtype)
    expected[:, :, 2] = 1.0
    torch.testing.assert_close(captured["channel_probs"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(captured["mask"], torch.ones((3, 1), dtype=torch.bool))


def test_rsample_raises_when_logits_shape_is_invalid(monkeypatch):
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=3,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = torch.full((2, 6), torch.nan)

    monkeypatch.setattr(
        module.root_node,
        "logits",
        torch.nn.Parameter(torch.zeros(1, module.n_root_nodes + 1, 1)),
    )
    with pytest.raises(InvalidParameterError):
        module.rsample(data=data)


def test_expectation_maximization_delegates_and_mle_is_unsupported(monkeypatch):
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=2,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )
    data = make_data(cls=Normal, out_features=6, n_samples=3)
    weights = torch.ones((3, 1, 1, 1))

    called = {"em": False}

    def fake_em(data_arg, bias_correction=True, *, cache=None):
        assert bias_correction is True
        called["em"] = True

    monkeypatch.setattr(module.root_node, "_expectation_maximization_step", fake_em)

    module._expectation_maximization_step(data, cache=Cache())
    with pytest.raises(AttributeError):
        module.maximum_likelihood_estimation(data, weights=weights)

    assert called["em"]


def test_marginalize_delegates_to_root_node(monkeypatch):
    module = make_rat_spn(
        leaf_cls=Normal,
        depth=1,
        n_region_nodes=2,
        num_leaves=2,
        num_repetitions=1,
        n_root_nodes=2,
        num_features=6,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )

    sentinel = object()

    def fake_marginalize(marg_rvs, prune=True, cache=None):
        return sentinel

    monkeypatch.setattr(module.root_node, "marginalize", fake_marginalize)
    assert module.marginalize([0, 1]) is sentinel
