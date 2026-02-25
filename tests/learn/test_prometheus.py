"""Prometheus implementation-specific helper/branch tests."""

from __future__ import annotations

import pytest
import torch

from spflow.learn import learn_prometheus
from spflow.learn.prometheus import (
    _adapt_product_inputs,
    _affinity_corr,
    _affinity_rdc,
    _cluster_by_kmeans,
    _create_partitioned_mv_leaf,
    _default_samples_per_var,
    _infer_scope_from_leaf_modules,
    _learn_prometheus_single_channel,
    _mst_partitions_from_affinity,
    _mst_partitions_from_sampled_edges,
    _sampled_edges_corr,
    _sampled_edges_rdc,
)
from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.products import Product
from spflow.modules.sums import Sum


def _randn(*size: int) -> torch.Tensor:
    return torch.randn(*size)


def _single_cluster(_data: torch.Tensor, n_clusters: int = 1) -> torch.Tensor:
    del n_clusters
    return torch.zeros((_data.shape[0],), dtype=torch.long, device=_data.device)


def test_prometheus_rejects_unknown_affinity_mode():
    data = _randn(50, 3)
    leaf = Normal(scope=Scope(list(range(3))), out_channels=1)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(data=data, leaf_modules=leaf, affinity_mode="full_unknown")


def test_prometheus_sampled_rejects_custom_similarity():
    data = _randn(50, 3)
    leaf = Normal(scope=Scope(list(range(3))), out_channels=1)

    def _custom_similarity(scoped_data: torch.Tensor) -> torch.Tensor:
        return torch.eye(scoped_data.shape[1], device=scoped_data.device, dtype=scoped_data.dtype)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(data=data, leaf_modules=leaf, similarity=_custom_similarity, affinity_mode="sampled")


def test_prometheus_rejects_invalid_sampling_per_var():
    data = _randn(50, 3)
    leaf = Normal(scope=Scope(list(range(3))), out_channels=1)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(data=data, leaf_modules=leaf, affinity_mode="sampled", sampling_per_var=0)


def test_prometheus_sampled_clamps_sampling_per_var():
    data = _randn(100, 4)
    leaf = Normal(scope=Scope(list(range(4))), out_channels=1)

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=20,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
        affinity_mode="sampled",
        sampling_per_var=10,
        sampling_seed=1,
    )

    # Oversized sampling budgets should be clamped internally, not rejected.
    assert isinstance(model, Module)


def test_cluster_by_kmeans_uses_preprocessing(monkeypatch):
    data = _randn(12, 3)
    seen: dict[str, torch.Tensor] = {}

    class _DummyKMeans:
        def __init__(self, n_clusters: int, mode: str, verbose: int):
            assert n_clusters == 3
            assert mode == "euclidean"
            assert verbose == 0

        def fit_predict(self, x: torch.Tensor) -> torch.Tensor:
            seen["x"] = x
            return torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)

    monkeypatch.setattr("spflow.learn.prometheus.KMeans", _DummyKMeans)
    labels = _cluster_by_kmeans(data, n_clusters=3, preprocessing=lambda x: x + 1.0)

    assert torch.all(labels == 0)
    assert "x" in seen
    assert torch.allclose(seen["x"], data + 1.0)


def test_cluster_by_kmeans_without_preprocessing(monkeypatch):
    data = _randn(9, 2)
    seen: dict[str, torch.Tensor] = {}

    class _DummyKMeans:
        def __init__(self, n_clusters: int, mode: str, verbose: int):
            assert n_clusters == 2
            assert mode == "euclidean"
            assert verbose == 0

        def fit_predict(self, x: torch.Tensor) -> torch.Tensor:
            seen["x"] = x
            return torch.ones((x.shape[0],), dtype=torch.long, device=x.device)

    monkeypatch.setattr("spflow.learn.prometheus.KMeans", _DummyKMeans)
    labels = _cluster_by_kmeans(data, n_clusters=2)

    assert torch.all(labels == 1)
    assert "x" in seen
    # Identity check guards the no-preprocessing fast path (no hidden copies/transforms).
    assert seen["x"] is data


def test_adapt_product_inputs_wraps_smaller_channels():
    low = Normal(scope=Scope([0]), out_channels=1)
    high = Normal(scope=Scope([1]), out_channels=3)

    adapted = _adapt_product_inputs([low, high])

    assert len(adapted) == 2
    assert isinstance(adapted[0], Sum)
    assert adapted[0].out_shape.channels == 3
    assert adapted[1] is high


def test_affinity_helpers_validate_and_handle_degenerate_cases():
    with pytest.raises(InvalidParameterError):
        _affinity_corr(_randn(3))
    with pytest.raises(UnsupportedOperationError):
        _affinity_corr(torch.tensor([[1.0], [float("nan")]]))
    assert torch.allclose(_affinity_corr(_randn(10, 1)), torch.ones(1, 1))

    with pytest.raises(InvalidParameterError):
        _affinity_rdc(_randn(3))
    with pytest.raises(UnsupportedOperationError):
        _affinity_rdc(torch.tensor([[1.0], [float("nan")]]))


def test_mst_partitions_affinity_validation_and_small_graph():
    with pytest.raises(InvalidParameterError):
        _mst_partitions_from_affinity(_randn(2, 3))
    assert _mst_partitions_from_affinity(torch.ones(1, 1)) == []


def test_default_samples_per_var_small_counts():
    # Tiny feature counts should avoid edge sampling entirely.
    assert _default_samples_per_var(0) == 0
    assert _default_samples_per_var(1) == 0
    assert _default_samples_per_var(8) >= 1


def test_sampled_edges_helpers_validate_and_handle_small_inputs():
    with pytest.raises(InvalidParameterError):
        _sampled_edges_corr(_randn(3), samples_per_var=1, seed=0)
    with pytest.raises(UnsupportedOperationError):
        _sampled_edges_corr(torch.tensor([[1.0], [float("nan")]]), samples_per_var=1, seed=0)
    assert _sampled_edges_corr(_randn(8, 1), samples_per_var=1, seed=0) == []

    with pytest.raises(InvalidParameterError):
        _sampled_edges_rdc(_randn(3), samples_per_var=1, seed=0)
    with pytest.raises(UnsupportedOperationError):
        _sampled_edges_rdc(torch.tensor([[1.0], [float("nan")]]), samples_per_var=1, seed=0)
    assert _sampled_edges_rdc(_randn(8, 1), samples_per_var=1, seed=0) == []


def test_mst_partitions_sampled_edges_disconnected_components():
    assert _mst_partitions_from_sampled_edges(1, []) == []

    partitions = _mst_partitions_from_sampled_edges(
        num_features=4,
        edges=[(0, 1, 1.0), (2, 3, 0.8)],
    )

    assert partitions
    # Two disjoint components need one bridge edge, yielding a 3-level merge schedule.
    assert len(partitions) == 3
    assert len(partitions[-1]) == 4


def test_infer_scope_from_leaf_modules_list_and_empty_validation():
    leaf_a = Normal(scope=Scope([0, 1]), out_channels=1)
    leaf_b = Normal(scope=Scope([2]), out_channels=1)

    merged = _infer_scope_from_leaf_modules([leaf_a, leaf_b])
    assert tuple(merged.query) == (0, 1, 2)

    with pytest.raises(InvalidParameterError):
        _infer_scope_from_leaf_modules([])


def test_create_partitioned_mv_leaf_handles_partitions_and_errors():
    data = _randn(64, 4)
    leaf_01 = Normal(scope=Scope([0, 1]), out_channels=1)
    leaf_23 = Normal(scope=Scope([2, 3]), out_channels=1)
    leaf_0 = Normal(scope=Scope([0]), out_channels=1)

    multi = _create_partitioned_mv_leaf(Scope([0, 1, 2, 3]), data, [leaf_01, leaf_23])
    assert isinstance(multi, Product)

    single_group = _create_partitioned_mv_leaf(Scope([0, 1]), data, [leaf_01, leaf_23])
    assert isinstance(single_group, Product)

    singleton = _create_partitioned_mv_leaf(Scope([0]), data, [leaf_0, leaf_23])
    assert isinstance(singleton, LeafModule)

    with pytest.raises(InvalidParameterError):
        _create_partitioned_mv_leaf(Scope([3]), data, [leaf_01])


def test_single_channel_falls_back_when_clusters_too_small():
    data = _randn(6, 3)
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=1)

    def _all_singletons(scoped_data: torch.Tensor, n_clusters: int = 2) -> torch.Tensor:
        del n_clusters
        return torch.arange(scoped_data.shape[0], dtype=torch.long, device=scoped_data.device)

    model = _learn_prometheus_single_channel(
        data=data,
        leaf_modules=[leaf],
        scope=Scope([0, 1, 2]),
        min_features_slice=1,
        min_instances_slice=1,
        n_clusters=2,
        clustering_method=_all_singletons,
        affinity_method=_affinity_corr,
        affinity_mode="full",
        similarity_kind="corr",
        sampling_per_var=None,
        sampling_seed=0,
    )

    assert isinstance(model, Product)


def test_single_channel_falls_back_when_no_partitions():
    data = _randn(12, 1)
    leaf = Normal(scope=Scope([0]), out_channels=1)

    model = _learn_prometheus_single_channel(
        data=data,
        leaf_modules=[leaf],
        scope=Scope([0]),
        min_features_slice=1,
        min_instances_slice=1,
        n_clusters=1,
        clustering_method=_single_cluster,
        affinity_method=_affinity_corr,
        affinity_mode="full",
        similarity_kind="corr",
        sampling_per_var=None,
        sampling_seed=0,
    )

    assert isinstance(model, LeafModule)


def test_single_channel_returns_single_product_when_only_one_partition():
    data = _randn(40, 2)
    leaf = Normal(scope=Scope([0, 1]), out_channels=1)

    model = _learn_prometheus_single_channel(
        data=data,
        leaf_modules=[leaf],
        scope=Scope([0, 1]),
        min_features_slice=2,
        min_instances_slice=1,
        n_clusters=1,
        clustering_method=_single_cluster,
        affinity_method=_affinity_corr,
        affinity_mode="full",
        similarity_kind="corr",
        sampling_per_var=None,
        sampling_seed=0,
    )

    assert isinstance(model, Product)


def test_single_channel_falls_back_when_total_weight_non_positive(monkeypatch):
    data = _randn(24, 2)
    leaf = Normal(scope=Scope([0, 1]), out_channels=1)

    class _ZeroChannelProduct:
        def __init__(self, inputs):
            self.inputs = inputs
            self.out_shape = type("Shape", (), {"channels": -1})()

    root_fallback = object()

    def _fake_create_partitioned_mv_leaf(scope: Scope, data: torch.Tensor, leaf_modules: list[LeafModule]):
        del data, leaf_modules
        if tuple(scope.query) == (0, 1):
            return root_fallback
        return leaf

    monkeypatch.setattr("spflow.learn.prometheus.Product", _ZeroChannelProduct)
    monkeypatch.setattr(
        "spflow.learn.prometheus._create_partitioned_mv_leaf", _fake_create_partitioned_mv_leaf
    )

    model = _learn_prometheus_single_channel(
        data=data,
        leaf_modules=[leaf],
        scope=Scope([0, 1]),
        min_features_slice=2,
        min_instances_slice=1,
        n_clusters=1,
        clustering_method=_single_cluster,
        affinity_method=_affinity_corr,
        affinity_mode="full",
        similarity_kind="corr",
        sampling_per_var=None,
        sampling_seed=0,
    )

    # Non-positive mixture weight sum must trigger deterministic fallback root.
    assert model is root_fallback
