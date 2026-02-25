from __future__ import annotations

import pytest
import torch

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.modules.leaves import Normal
from spflow.modules.ops import SplitMode
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.test_helpers.builders import make_rat_spn
from tests.utils.leaves import make_data


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
    with pytest.raises(UnsupportedOperationError):
        module.log_posterior(make_data(cls=Normal, out_features=6, n_samples=3))


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
        module.root_node, "logits", torch.nn.Parameter(torch.zeros(1, module.n_root_nodes, 1))
    )

    captured: dict[str, SamplingContext] = {}
    original_sample = module.root_node.inputs._sample

    def capture_sample(*args, **kwargs):
        captured["ctx"] = kwargs["sampling_ctx"]
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(module.root_node.inputs, "_sample", capture_sample)

    # The public sample() API should bootstrap channel_index for both deterministic
    # and stochastic paths before delegating into the input sampler.
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
    monkeypatch.setattr(
        module.root_node, "logits", torch.nn.Parameter(torch.zeros(1, module.n_root_nodes + 1, 1))
    )
    # Root logits must align with n_root_nodes so class routing is well-defined.
    with pytest.raises(InvalidParameterError):
        module.sample(data=torch.full((2, 6), torch.nan))


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

    # MLE intentionally remains unsupported on this wrapper; EM still has to pass through.
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
