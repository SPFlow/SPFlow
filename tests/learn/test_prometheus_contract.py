"""Contracts for public learn_prometheus behavior across affinity/sampling modes."""

from __future__ import annotations

import pytest
import torch

from spflow.learn import learn_prometheus
from spflow.exceptions import InvalidParameterError, InvalidTypeError
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum


def _randn(*size: int) -> torch.Tensor:
    return torch.randn(*size)


def _single_cluster(_data: torch.Tensor, n_clusters: int = 1) -> torch.Tensor:
    del n_clusters
    return torch.zeros((_data.shape[0],), dtype=torch.long, device=_data.device)


@pytest.mark.contract
def test_prometheus_smoke():
    data = _randn(400, 4)
    scope = Scope(list(range(4)))
    leaf = Normal(scope=scope, out_channels=2)

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=50,
        n_clusters=2,
        similarity="corr",
    )

    assert isinstance(model, Module)
    assert tuple(model.scope.query) == tuple(range(4))
    assert torch.isfinite(model.log_likelihood(data[:8])).all()


@pytest.mark.contract
@pytest.mark.parametrize("affinity_mode", ["full", "sampled"])
@pytest.mark.parametrize("similarity", ["corr", "rdc"])
@pytest.mark.parametrize("sampling_seed", [None, 7])
@pytest.mark.parametrize("sampling_per_var", [None, 1])
def test_prometheus_options_smoke(affinity_mode, similarity, sampling_seed, sampling_per_var):
    data = _randn(120, 5)
    leaf = Normal(scope=Scope(list(range(5))), out_channels=1)

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity=similarity,
        affinity_mode=affinity_mode,
        sampling_per_var=sampling_per_var,
        sampling_seed=sampling_seed,
    )

    assert isinstance(model, Module)
    assert torch.isfinite(model.log_likelihood(data[:8])).all()


@pytest.mark.contract
def test_prometheus_reuses_scopes_within_cluster():
    base = _randn(300, 1)
    data = torch.cat(
        [base + 0.05 * _randn(300, 1), base + 0.05 * _randn(300, 1), _randn(300, 1), _randn(300, 1)], dim=1
    ).float()

    model = learn_prometheus(
        data=data,
        leaf_modules=Normal(scope=Scope(list(range(4))), out_channels=1),
        out_channels=1,
        min_instances_slice=2,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
    )

    # Root can differ by optimization path; both layouts should expose product candidates.
    if isinstance(model, Sum):
        products_cat = model.inputs
    elif isinstance(model, Cat) and all(isinstance(m, Sum) for m in model.inputs):
        products_cat = model.inputs[0].inputs
    else:
        raise AssertionError(f"Unexpected model root type: {type(model)}")

    assert isinstance(products_cat, Cat)
    product_nodes = list(products_cat.inputs)
    assert len(product_nodes) >= 2
    assert all(isinstance(p, Product) for p in product_nodes)


@pytest.mark.contract
@pytest.mark.parametrize("num_features", [1, 2])
def test_prometheus_sampled_small_feature_counts(num_features: int):
    data = _randn(80, num_features)
    scope = Scope(list(range(num_features)))
    leaf = Normal(scope=scope, out_channels=1)

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=2,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="sampled",
        sampling_seed=0,
    )

    assert isinstance(model, Module)
    assert model.scope.query == scope.query
    if num_features == 1:
        assert isinstance(model, LeafModule)
    assert torch.isfinite(model.log_likelihood(data[:8])).all()


@pytest.mark.contract
def test_prometheus_sampled_deterministic_with_seed():
    data = _randn(140, 6)
    leaf = Normal(scope=Scope(list(range(6))), out_channels=1)

    model_a = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="sampled",
        sampling_seed=42,
    )
    model_b = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="sampled",
        sampling_seed=42,
    )

    assert torch.allclose(model_a.log_likelihood(data[:8]), model_b.log_likelihood(data[:8]))


@pytest.mark.contract
def test_prometheus_sampled_complete_matches_full_corr():
    data = _randn(160, 5)
    leaf = Normal(scope=Scope(list(range(5))), out_channels=1)

    model_full = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="full",
    )
    model_sampled = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="sampled",
        sampling_per_var=4,
        sampling_seed=123,
    )

    # With enough sampled edges, sampled affinity should match full-correlation behavior.
    assert torch.allclose(model_full.log_likelihood(data[:8]), model_sampled.log_likelihood(data[:8]))


@pytest.mark.contract
def test_prometheus_sampled_rdc_deterministic_with_seed():
    data = _randn(90, 4)
    leaf = Normal(scope=Scope(list(range(4))), out_channels=1)

    # Reset global RNG to isolate determinism guarantees to the explicit sampling seed.
    torch.manual_seed(0)
    model_a = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=10,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="rdc",
        affinity_mode="sampled",
        sampling_seed=3,
    )
    torch.manual_seed(0)
    model_b = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=10,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="rdc",
        affinity_mode="sampled",
        sampling_seed=3,
    )

    assert torch.allclose(model_a.log_likelihood(data[:6]), model_b.log_likelihood(data[:6]))


@pytest.mark.contract
def test_prometheus_applies_partial_args_and_multi_channel_output():
    data = _randn(80, 3)
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=1)

    def _cluster_with_shift(scoped_data: torch.Tensor, n_clusters: int = 1, shift: int = 0) -> torch.Tensor:
        del n_clusters
        return torch.full((scoped_data.shape[0],), shift, dtype=torch.long, device=scoped_data.device)

    def _constant_affinity(scoped_data: torch.Tensor, fill: float = 0.5) -> torch.Tensor:
        d = scoped_data.shape[1]
        a = torch.full((d, d), fill, device=scoped_data.device, dtype=scoped_data.dtype)
        a.fill_diagonal_(1.0)
        return a

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=2,
        min_instances_slice=2,
        n_clusters=1,
        clustering_method=_cluster_with_shift,
        similarity=_constant_affinity,
        clustering_args={"shift": 0},
        similarity_args={"fill": 0.7},
    )

    # Multi-channel output is represented as concatenated per-channel submodels.
    assert isinstance(model, Cat)
    assert torch.isfinite(model.log_likelihood(data[:5])).all()


@pytest.mark.contract
def test_prometheus_infers_scope_from_leaf_module_list():
    data = _randn(64, 3)
    model = learn_prometheus(
        data=data,
        leaf_modules=[Normal(scope=Scope([0, 1]), out_channels=1), Normal(scope=Scope([2]), out_channels=1)],
        out_channels=1,
        min_instances_slice=1000,
    )
    assert isinstance(model, Module)
    assert tuple(model.scope.query) == (0, 1, 2)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("kwargs", "error_type", "error_match"),
    [
        (
            {"leaf_modules": object()},
            InvalidTypeError,
            "'leaf_modules' must be a LeafModule or list\\[LeafModule\\]",
        ),
        ({"out_channels": 0}, InvalidParameterError, "'out_channels' must be >= 1"),
        ({"min_features_slice": 0}, InvalidParameterError, "'min_features_slice' must be >= 1"),
        ({"min_instances_slice": 0}, InvalidParameterError, "'min_instances_slice' must be >= 1"),
        ({"n_clusters": 0}, InvalidParameterError, "'n_clusters' must be >= 1"),
        ({"clustering_method": "unknown"}, InvalidParameterError, "Unknown clustering_method"),
        ({"similarity": "unknown"}, InvalidParameterError, "Unknown similarity"),
    ],
)
def test_prometheus_parameter_validation_contract(kwargs, error_type, error_match):
    data = _randn(40, 3)
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=1)

    params = {"data": data, "leaf_modules": leaf, "scope": Scope([0, 1, 2])}
    params.update(kwargs)
    with pytest.raises(error_type):
        learn_prometheus(**params)
