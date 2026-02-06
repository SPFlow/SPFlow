import pytest
import torch

from spflow.learn import learn_prometheus
from spflow.exceptions import InvalidParameterError, InvalidTypeError, UnsupportedOperationError
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
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum


def _single_cluster(_data: torch.Tensor, n_clusters: int = 1) -> torch.Tensor:
    del n_clusters
    return torch.zeros((_data.shape[0],), dtype=torch.long, device=_data.device)


def test_prometheus_smoke():
    data = torch.randn(400, 4)
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

    lls = model.log_likelihood(data[:8])
    assert torch.isfinite(lls).all()


def test_prometheus_reuses_scopes_within_cluster():
    # Create mildly correlated data so the MST is non-degenerate, but the exact ordering does not matter.
    base = torch.randn(300, 1)
    data = torch.cat(
        [
            base + 0.05 * torch.randn(300, 1),
            base + 0.05 * torch.randn(300, 1),
            torch.randn(300, 1),
            torch.randn(300, 1),
        ],
        dim=1,
    ).float()

    scope = Scope(list(range(4)))
    leaf = Normal(scope=scope, out_channels=1)

    model = learn_prometheus(
        data=data,
        leaf_modules=leaf,
        out_channels=1,
        min_instances_slice=2,
        n_clusters=1,
        clustering_method=_single_cluster,
        similarity="corr",
    )

    # Find all product nodes directly under the root mixture.
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

    # Collect occurrences of child scopes and verify at least one repeated scope is the same object instance.
    scope_to_ids: dict[tuple[int, ...], list[int]] = {}
    for product in product_nodes:
        child = product.inputs
        if isinstance(child, Cat):
            children = list(child.inputs)
        else:
            children = [child]
        for c in children:
            key = tuple(c.scope.query)
            scope_to_ids.setdefault(key, []).append(id(c))

    shared = [ids for ids in scope_to_ids.values() if len(ids) >= 2 and len(set(ids)) == 1]
    assert shared, "Expected at least one reused sub-scope module instance across product decompositions."


@pytest.mark.parametrize("affinity_mode", ["full", "sampled"])
@pytest.mark.parametrize("similarity", ["corr", "rdc"])
@pytest.mark.parametrize("sampling_seed", [None, 7])
@pytest.mark.parametrize("sampling_per_var", [None, 1])
def test_prometheus_options_smoke(
    affinity_mode: str,
    similarity: str,
    sampling_seed: int | None,
    sampling_per_var: int | None,
):
    data = torch.randn(120, 5)
    scope = Scope(list(range(5)))
    leaf = Normal(scope=scope, out_channels=1)

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
    lls = model.log_likelihood(data[:8])
    assert torch.isfinite(lls).all()


@pytest.mark.parametrize("affinity_mode", ["full", "sampled"])
def test_prometheus_rejects_unknown_affinity_mode(affinity_mode: str):
    data = torch.randn(50, 3)
    scope = Scope(list(range(3)))
    leaf = Normal(scope=scope, out_channels=1)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(
            data=data,
            leaf_modules=leaf,
            affinity_mode=affinity_mode + "_unknown",
        )


def test_prometheus_sampled_rejects_custom_similarity():
    data = torch.randn(50, 3)
    scope = Scope(list(range(3)))
    leaf = Normal(scope=scope, out_channels=1)

    def _custom_similarity(scoped_data: torch.Tensor) -> torch.Tensor:
        return torch.eye(scoped_data.shape[1], device=scoped_data.device, dtype=scoped_data.dtype)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(
            data=data,
            leaf_modules=leaf,
            similarity=_custom_similarity,
            affinity_mode="sampled",
        )


def test_prometheus_rejects_invalid_sampling_per_var():
    data = torch.randn(50, 3)
    scope = Scope(list(range(3)))
    leaf = Normal(scope=scope, out_channels=1)

    with pytest.raises(InvalidParameterError):
        learn_prometheus(
            data=data,
            leaf_modules=leaf,
            affinity_mode="sampled",
            sampling_per_var=0,
        )


def test_prometheus_sampled_deterministic_with_seed():
    data = torch.randn(140, 6)
    scope = Scope(list(range(6)))
    leaf = Normal(scope=scope, out_channels=1)

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

    ll_a = model_a.log_likelihood(data[:8])
    ll_b = model_b.log_likelihood(data[:8])
    assert torch.allclose(ll_a, ll_b)


def test_prometheus_sampled_complete_matches_full_corr():
    data = torch.randn(160, 5)
    scope = Scope(list(range(5)))
    leaf = Normal(scope=scope, out_channels=1)

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

    ll_full = model_full.log_likelihood(data[:8])
    ll_sampled = model_sampled.log_likelihood(data[:8])
    assert torch.allclose(ll_full, ll_sampled)


@pytest.mark.parametrize("num_features", [1, 2])
def test_prometheus_sampled_small_feature_counts(num_features: int):
    data = torch.randn(80, num_features)
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
    lls = model.log_likelihood(data[:8])
    assert torch.isfinite(lls).all()


def test_prometheus_sampled_clamps_sampling_per_var():
    data = torch.randn(100, 4)
    scope = Scope(list(range(4)))
    leaf = Normal(scope=scope, out_channels=1)

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

    assert isinstance(model, Module)


def test_prometheus_sampled_rdc_deterministic_with_seed():
    data = torch.randn(90, 4)
    scope = Scope(list(range(4)))
    leaf = Normal(scope=scope, out_channels=1)

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

    ll_a = model_a.log_likelihood(data[:6])
    ll_b = model_b.log_likelihood(data[:6])
    assert torch.allclose(ll_a, ll_b)


def test_cluster_by_kmeans_uses_preprocessing(monkeypatch):
    data = torch.randn(12, 3)
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
    data = torch.randn(9, 2)
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
        _affinity_corr(torch.randn(3))
    with pytest.raises(UnsupportedOperationError):
        _affinity_corr(torch.tensor([[1.0], [float("nan")]]))
    assert torch.allclose(_affinity_corr(torch.randn(10, 1)), torch.ones(1, 1))

    with pytest.raises(InvalidParameterError):
        _affinity_rdc(torch.randn(3))
    with pytest.raises(UnsupportedOperationError):
        _affinity_rdc(torch.tensor([[1.0], [float("nan")]]))


def test_mst_partitions_affinity_validation_and_small_graph():
    with pytest.raises(InvalidParameterError):
        _mst_partitions_from_affinity(torch.randn(2, 3))
    assert _mst_partitions_from_affinity(torch.ones(1, 1)) == []


def test_default_samples_per_var_small_counts():
    assert _default_samples_per_var(0) == 0
    assert _default_samples_per_var(1) == 0
    assert _default_samples_per_var(8) >= 1


def test_sampled_edges_helpers_validate_and_handle_small_inputs():
    with pytest.raises(InvalidParameterError):
        _sampled_edges_corr(torch.randn(3), samples_per_var=1, seed=0)
    with pytest.raises(UnsupportedOperationError):
        _sampled_edges_corr(torch.tensor([[1.0], [float("nan")]]), samples_per_var=1, seed=0)
    assert _sampled_edges_corr(torch.randn(8, 1), samples_per_var=1, seed=0) == []

    with pytest.raises(InvalidParameterError):
        _sampled_edges_rdc(torch.randn(3), samples_per_var=1, seed=0)
    with pytest.raises(UnsupportedOperationError):
        _sampled_edges_rdc(torch.tensor([[1.0], [float("nan")]]), samples_per_var=1, seed=0)
    assert _sampled_edges_rdc(torch.randn(8, 1), samples_per_var=1, seed=0) == []


def test_mst_partitions_sampled_edges_disconnected_components():
    assert _mst_partitions_from_sampled_edges(1, []) == []

    partitions = _mst_partitions_from_sampled_edges(
        num_features=4,
        edges=[(0, 1, 1.0), (2, 3, 0.8)],
    )

    assert partitions
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
    data = torch.randn(64, 4)
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
    data = torch.randn(6, 3)
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
    data = torch.randn(12, 1)
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
    data = torch.randn(40, 2)
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
    data = torch.randn(24, 2)
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
    monkeypatch.setattr("spflow.learn.prometheus._create_partitioned_mv_leaf", _fake_create_partitioned_mv_leaf)

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

    assert model is root_fallback


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"leaf_modules": object()}, InvalidTypeError),
        ({"out_channels": 0}, InvalidParameterError),
        ({"min_features_slice": 0}, InvalidParameterError),
        ({"min_instances_slice": 0}, InvalidParameterError),
        ({"n_clusters": 0}, InvalidParameterError),
        ({"clustering_method": "unknown"}, InvalidParameterError),
        ({"similarity": "unknown"}, InvalidParameterError),
    ],
)
def test_prometheus_parameter_validation_branches(kwargs, error_type):
    data = torch.randn(40, 3)
    leaf = Normal(scope=Scope([0, 1, 2]), out_channels=1)

    params = {"data": data, "leaf_modules": leaf, "scope": Scope([0, 1, 2])}
    params.update(kwargs)
    with pytest.raises(error_type):
        learn_prometheus(**params)


def test_prometheus_applies_partial_args_and_multi_channel_output():
    data = torch.randn(80, 3)
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

    assert isinstance(model, Cat)
    lls = model.log_likelihood(data[:5])
    assert torch.isfinite(lls).all()


def test_prometheus_infers_scope_from_leaf_module_list():
    data = torch.randn(64, 3)
    leaf_01 = Normal(scope=Scope([0, 1]), out_channels=1)
    leaf_2 = Normal(scope=Scope([2]), out_channels=1)

    model = learn_prometheus(
        data=data,
        leaf_modules=[leaf_01, leaf_2],
        out_channels=1,
        min_instances_slice=1000,
    )

    assert isinstance(model, Module)
    assert tuple(model.scope.query) == (0, 1, 2)
