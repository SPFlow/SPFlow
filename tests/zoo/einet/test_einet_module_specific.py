"""Einet implementation-specific branches not covered by cross-model contracts."""

from __future__ import annotations

from itertools import product

import pytest
import torch
from torch import nn

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.contract_data import EINET_LAYER_TYPE_VALUES, EINET_STRUCTURE_VALUES
from tests.test_helpers.builders import make_einet, make_einet_leaf_modules


def test_invalid_depth():
    with pytest.raises(ValueError):
        make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=10, num_repetitions=2)


def test_invalid_layer_type():
    with pytest.raises(ValueError):
        make_einet(
            num_features=4,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            layer_type="invalid",
        )


def test_invalid_structure():
    with pytest.raises(ValueError):
        make_einet(
            num_features=4,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            structure="invalid",
        )


@pytest.mark.parametrize("layer_type,structure", product(EINET_LAYER_TYPE_VALUES, EINET_STRUCTURE_VALUES))
def test_gradient_flow(layer_type: str, structure: str):
    model = make_einet(
        num_features=4,
        num_classes=1,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
        layer_type=layer_type,
        structure=structure,
    )

    loss = -model.log_likelihood(torch.randn(20, 4)).mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize("layer_type,structure", product(EINET_LAYER_TYPE_VALUES, EINET_STRUCTURE_VALUES))
def test_optimization(layer_type: str, structure: str):
    model = make_einet(
        num_features=4,
        num_classes=1,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
        layer_type=layer_type,
        structure=structure,
    )

    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    data = torch.randn(50, 4)

    for _ in range(5):
        optimizer.zero_grad()
        loss = -model.log_likelihood(data).mean()
        loss.backward()
        optimizer.step()

    # Exact inequality (zero tolerances) ensures we catch no-op optimizer updates.
    assert any(
        not torch.allclose(param, initial_params[name], rtol=0.0, atol=0.0)
        for name, param in model.named_parameters()
    )


@pytest.mark.parametrize("layer_type,structure", product(EINET_LAYER_TYPE_VALUES, EINET_STRUCTURE_VALUES))
def test_extra_repr(layer_type: str, structure: str):
    model = make_einet(
        num_features=4,
        num_classes=2,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
        layer_type=layer_type,
        structure=structure,
    )
    repr_str = model.extra_repr()
    assert "num_features=4" in repr_str
    assert "num_classes=2" in repr_str
    assert f"layer_type={layer_type}" in repr_str
    assert f"structure={structure}" in repr_str


def test_bottom_up_sampling_not_implemented():
    model = make_einet(
        num_features=4,
        num_classes=1,
        num_sums=5,
        num_leaves=3,
        depth=1,
        num_repetitions=2,
        structure="bottom-up",
    )
    with pytest.raises(NotImplementedError):
        model.sample(num_samples=10)


def test_more_invalid_constructor_parameters():
    leaf_modules = make_einet_leaf_modules(4, 3, 1)
    from spflow.zoo.einet import Einet

    with pytest.raises(ValueError):
        Einet(leaf_modules=leaf_modules, num_classes=0)
    with pytest.raises(ValueError):
        Einet(leaf_modules=leaf_modules, num_sums=0)
    with pytest.raises(ValueError):
        Einet(leaf_modules=leaf_modules, num_leaves=0)
    with pytest.raises(ValueError):
        Einet(leaf_modules=leaf_modules, depth=-1)
    with pytest.raises(ValueError):
        Einet(leaf_modules=leaf_modules, num_repetitions=0)


def test_properties_delegate_to_root():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    assert model.n_out == 1
    assert model.feature_to_scope.shape == model.root_node.feature_to_scope.shape
    with pytest.raises(AttributeError):
        _ = type(model).scopes_out.fget(model)


def test_log_posterior_raises_for_single_class():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    with pytest.raises(UnsupportedOperationError):
        model.log_posterior(torch.randn(3, 4))


def test_log_posterior_and_predict_proba_multiclass():
    model = make_einet(num_features=4, num_classes=3, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    data = torch.randn(5, 4)

    log_post = model.log_posterior(data)
    proba = model.predict_proba(data)

    assert log_post.shape == (5, 3)
    assert proba.shape == (5, 3)
    assert torch.isfinite(log_post).all()
    assert torch.allclose(proba.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_multiclass_sample_mpe_and_stochastic():
    model = make_einet(num_features=4, num_classes=3, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    data = torch.full((6, 4), torch.nan)

    class DummyInput(nn.Module):
        def __init__(self, num_features: int):
            super().__init__()
            self.num_features = num_features

        def _sample(self, data, sampling_ctx, cache):
            del sampling_ctx, cache
            return torch.zeros((data.shape[0], self.num_features))

    class DummyRoot(nn.Module):
        def __init__(self, num_classes: int, num_features: int):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros((1, num_classes, 1)))
            self.inputs = DummyInput(num_features)

    model.root_node = DummyRoot(model.num_classes, model.num_features)

    mpe_samples = model.sample(data=data, is_mpe=True)
    random_samples = model.sample(data=data, is_mpe=False)

    assert mpe_samples.shape == (6, 4)
    assert random_samples.shape == (6, 4)
    assert torch.isfinite(mpe_samples).all()
    assert torch.isfinite(random_samples).all()


def test_multiclass_sampling_raises_for_invalid_logits_shape():
    model = make_einet(num_features=4, num_classes=3, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    model.root_node.logits = torch.nn.Parameter(torch.zeros(2, 2))

    # The sampler expects class logits in [1, C, 1]; a rank-2 tensor should fail fast.
    with pytest.raises(InvalidParameterError):
        model.sample(num_samples=2)


def test_sample_defaults_to_one_sample():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    assert model.sample().shape == (1, 4)


def test_sample_initializes_repetition_index_for_single_rep():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    sampling_ctx = SamplingContext(num_samples=4)
    data = torch.full((4, 4), torch.nan)

    # Even with one repetition, downstream nodes index into repetition tensors.
    # Initialize to zeros to keep that invariant uniform across repetition counts.
    samples = model._sample(data=data, sampling_ctx=sampling_ctx, cache=Cache())
    assert samples.shape == (4, 4)
    assert sampling_ctx.repetition_index is not None
    assert torch.equal(sampling_ctx.repetition_index, torch.zeros(4, dtype=torch.long))


def test_sample_validates_internal_sampling_context():
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.zeros((2,), dtype=torch.long),
    )
    # Deliberately violate the internal invariant to exercise explicit shape validation.
    sampling_ctx._mask = torch.ones((2, 2), dtype=torch.bool)  # type: ignore[attr-defined]
    with pytest.raises(InvalidParameterError, match="mismatched channel_index/mask shapes"):
        model._sample(data=torch.full((2, 4), torch.nan), sampling_ctx=sampling_ctx, cache=Cache())


def test_delegating_methods(monkeypatch: pytest.MonkeyPatch):
    model = make_einet(num_features=4, num_classes=1, num_sums=5, num_leaves=3, depth=1, num_repetitions=1)
    data = torch.randn(3, 4)
    cache = Cache()

    calls = {"em": 0, "marginalize": 0}
    expected_result = object()

    def fake_em(call_data, bias_correction=True, *, cache=None):
        assert call_data is data
        assert bias_correction is True
        assert cache is cache_obj
        calls["em"] += 1

    def fake_marginalize(marg_rvs, prune=True, cache=None):
        assert marg_rvs == [0, 2]
        assert prune is False
        assert cache is cache_obj
        calls["marginalize"] += 1
        return expected_result

    cache_obj = cache
    monkeypatch.setattr(model.root_node, "_expectation_maximization_step", fake_em)
    monkeypatch.setattr(model.root_node, "marginalize", fake_marginalize)

    model._expectation_maximization_step(data, cache=cache_obj)
    with pytest.raises(AttributeError):
        model.maximum_likelihood_estimation(data)
    result = model.marginalize([0, 2], prune=False, cache=cache_obj)

    assert calls["em"] == 1
    assert calls["marginalize"] == 1
    assert result is expected_result
