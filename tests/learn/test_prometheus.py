import pytest
import torch

from spflow.learn import learn_prometheus
from spflow.exceptions import InvalidParameterError
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
