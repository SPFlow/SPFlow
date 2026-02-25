import importlib
from itertools import combinations, product
import pytest

import numpy as np
import torch

from spflow.learn import learn_spn
from spflow.learn.learn_spn import cluster_by_kmeans, partition_by_rdc, prune_sums
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.rdc import rdc

learn_spn_module = importlib.import_module("spflow.learn.learn_spn")


def _randn(*size, **kwargs) -> torch.Tensor:
    return torch.randn(*size, **kwargs)


def _rand(*size, **kwargs) -> torch.Tensor:
    return torch.rand(*size, **kwargs)


def test_kmeans():
    # Well-separated means reduce flakiness in unsupervised partition checks.

    cluster_1 = _randn((100, 1)) - 20.0
    cluster_2 = _randn((100, 1)) - 10.0
    cluster_3 = _randn((100, 1))
    cluster_4 = _randn((100, 1)) + 10.0
    cluster_5 = _randn((100, 1)) + 20.0

    cluster_mask = cluster_by_kmeans(
        torch.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]).float(), n_clusters=5
    )

    assert len(torch.unique(cluster_mask)) == 5


def make_rdc_data(n_samples=1000):
    # Mixed marginals make RDC partitioning handle non-Gaussian feature types.
    feature1 = _randn(n_samples)
    feature2 = _rand(n_samples) * 4 - 2
    feature3 = torch.distributions.Exponential(1.0).sample((n_samples,))
    feature4 = torch.distributions.Binomial(10, 0.5).sample((n_samples,))

    data = torch.stack((feature1, feature2, feature3, feature4), dim=1)
    return data


def test_rdc():
    from networkx import connected_components as ccnp, from_numpy_array

    data = make_rdc_data()
    threshold = 0.3

    # Build the feature dependency graph explicitly to validate threshold partitioning.
    rdcs = torch.eye(data.shape[1])
    for i, j in combinations(range(data.shape[1]), 2):
        r = rdc(data[:, i], data[:, j])
        rdcs[j][i] = rdcs[i][j] = r

    rdcs[rdcs < threshold] = 0.0
    adj_mat = rdcs

    partition_ids = torch.zeros(data.shape[1], dtype=torch.int)

    np_matrix = np.array(adj_mat.cpu().tolist())
    fna_matrix = from_numpy_array(np_matrix)

    for i, c in enumerate(ccnp(fna_matrix)):
        partition_ids[list(c)] = i + 1

    partition_ids

    partitions = []

    for partition_id in torch.sort(torch.unique(partition_ids), dim=-1)[0]:
        partitions.append(torch.where(partition_ids == partition_id))

    assert len(partitions) == 4


@pytest.mark.parametrize(
    "leaf_channel,sum_channel",
    list(product([1, 2], [1, 2])),
)
def test_multiple_features(leaf_channel, sum_channel):
    scope = Scope(list(range(5)))
    leaf_layer = Normal(scope=scope, out_channels=leaf_channel)

    # Diverse clusters make recursive split/cluster branches more likely to trigger.
    cluster_1 = _randn(200, 5) + torch.tensor([0, 0, 0, 0, 0])
    cluster_2 = _randn(200, 5) + torch.tensor([5, 5, 5, 5, 5])
    cluster_3 = _randn(200, 5) + torch.tensor([-5, -5, -5, -5, -5])
    cluster_4 = _randn(200, 5) + torch.tensor([10, 0, -10, 5, -5])
    cluster_5 = _randn(200, 5) + torch.tensor([-10, 5, 10, -5, 0])
    data = torch.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]).float()
    model = learn_spn(
        data,
        leaf_modules=leaf_layer,
        out_channels=sum_channel,
        min_instances_slice=100,
    )

    assert isinstance(model, Module)
    assert tuple(model.scope.query) == tuple(range(5))

    batch = 8
    lls = model.log_likelihood(data[:batch])
    assert lls.shape == (
        batch,
        model.out_shape.features,
        model.out_shape.channels,
        model.out_shape.repetitions,
    )
    assert torch.isfinite(lls).all()


def test_make_moons():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.datasets import make_moons  # noqa: F401

    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    scope = Scope(list(range(2)))
    normal_layer = Normal(scope=scope, out_channels=4)

    spn = learn_spn(
        torch.tensor(X, dtype=torch.float32),
        leaf_modules=normal_layer,
        out_channels=1,
        min_instances_slice=70,
    )
    num_params = sum(p.numel() for p in spn.parameters() if p.requires_grad)
    prune_sums(spn)
    num_params_after_pruning = sum(p.numel() for p in spn.parameters() if p.requires_grad)
    assert num_params_after_pruning <= num_params
    lls = spn.log_likelihood(torch.tensor(X[:8], dtype=torch.float32))
    assert torch.isfinite(lls).all()


def test_prune_sums_flattens_nested_sums():
    scope = Scope([0])
    leaf1 = Normal(scope=scope, out_channels=1)
    leaf2 = Normal(scope=scope, out_channels=1)

    child_sum1 = Sum(inputs=leaf1, out_channels=1)
    child_sum2 = Sum(inputs=leaf2, out_channels=1)
    root_sum = Sum(inputs=[child_sum1, child_sum2], out_channels=1)

    data = _randn(16, 1)
    lls_before = root_sum.log_likelihood(data)
    num_sums_before = sum(1 for m in root_sum.modules() if isinstance(m, Sum))

    prune_sums(root_sum)

    lls_after = root_sum.log_likelihood(data)
    num_sums_after = sum(1 for m in root_sum.modules() if isinstance(m, Sum))

    assert num_sums_after < num_sums_before
    assert torch.allclose(lls_before, lls_after)


def test_partition_by_rdc_applies_preprocessing_and_restores_dtype():
    data = _randn(24, 3, dtype=torch.float32)

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        was_called = {"value": False}

        def preprocessing(x):
            was_called["value"] = True
            return x + 0.123

        partition_ids = partition_by_rdc(data, threshold=1.1, preprocessing=preprocessing)

        assert was_called["value"] is True
        assert partition_ids.shape == (data.shape[1],)
        assert partition_ids.device == data.device
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(original_dtype)


def test_cluster_by_kmeans_applies_preprocessing(monkeypatch):
    seen = {}

    class DummyKMeans:
        def __init__(self, n_clusters, mode, verbose):
            seen["args"] = (n_clusters, mode, verbose)

        def fit_predict(self, data):
            seen["data"] = data
            return torch.zeros(data.shape[0], dtype=torch.int64)

    monkeypatch.setattr(learn_spn_module, "KMeans", DummyKMeans)

    data = _randn(9, 2)
    labels = cluster_by_kmeans(data, n_clusters=3, preprocessing=lambda x: x + 1.0)

    assert seen["args"] == (3, "euclidean", 1)
    assert torch.equal(seen["data"], data + 1.0)
    assert labels.shape == (data.shape[0],)


def test_learn_spn_builds_scope_from_disjoint_leaf_list_and_returns_product():
    data = _randn(16, 2)
    leaves = [Normal(scope=Scope([0]), out_channels=1), Normal(scope=Scope([1]), out_channels=1)]

    model = learn_spn(data, leaf_modules=leaves, min_features_slice=3)

    assert isinstance(model, Product)
    assert tuple(model.scope.query) == (0, 1)


def test_learn_spn_single_leaf_list_uses_single_scope_path():
    data = _randn(16, 1)
    leaves = [Normal(scope=Scope([0]), out_channels=1)]

    model = learn_spn(data, leaf_modules=leaves, min_features_slice=3)

    assert isinstance(model, Normal)
    assert tuple(model.scope.query) == (0,)


def test_learn_spn_rejects_non_disjoint_leaf_scopes():
    data = _randn(16, 3)
    leaves = [Normal(scope=Scope([0, 1]), out_channels=1), Normal(scope=Scope([1, 2]), out_channels=1)]

    with pytest.raises(ValueError):
        learn_spn(data, leaf_modules=leaves)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"clustering_method": "invalid"}, "invalid"),
        ({"partitioning_method": "invalid"}, "invalid"),
        ({"min_instances_slice": 1}, "min_instances_slice"),
        ({"min_features_slice": 1}, "min_features_slice"),
    ],
)
def test_learn_spn_validates_inputs(kwargs, match):
    data = _randn(8, 1)
    leaf = Normal(scope=Scope([0]), out_channels=1)

    with pytest.raises(ValueError):
        learn_spn(data, leaf_modules=leaf, **kwargs)


def test_learn_spn_applies_method_kwargs_via_partial_binding():
    data = _randn(20, 2)
    leaf = Normal(scope=Scope([0, 1]), out_channels=1)
    seen = {"partition_tokens": [], "cluster_tokens": []}

    def partitioning_method(part_data, token):
        seen["partition_tokens"].append(token)
        return torch.tensor([0, 1], device=part_data.device)

    def clustering_method(cluster_data, token):
        seen["cluster_tokens"].append(token)
        return torch.zeros(cluster_data.shape[0], dtype=torch.int64, device=cluster_data.device)

    model = learn_spn(
        data,
        leaf_modules=leaf,
        partitioning_method=partitioning_method,
        clustering_method=clustering_method,
        partitioning_args={"token": "partition-token"},
        clustering_args={"token": "cluster-token"},
    )

    assert isinstance(model, Product)
    assert seen["partition_tokens"][0] == "partition-token"


def test_learn_spn_single_cluster_keeps_inputs_as_list_branch():
    data = _randn(30, 2)
    leaf = Normal(scope=Scope([0, 1]), out_channels=1)
    state = {"partition_calls": 0}

    def partitioning_method(part_data):
        state["partition_calls"] += 1
        if state["partition_calls"] == 1:
            return torch.tensor([1, 1], device=part_data.device)
        return torch.tensor([1, 2], device=part_data.device)

    def clustering_method(cluster_data):
        return torch.zeros(cluster_data.shape[0], dtype=torch.int64, device=cluster_data.device)

    model = learn_spn(
        data,
        leaf_modules=leaf,
        partitioning_method=partitioning_method,
        clustering_method=clustering_method,
        min_instances_slice=2,
        min_features_slice=2,
        out_channels=1,
    )

    assert isinstance(model, Sum)
    assert state["partition_calls"] >= 2


def test_learn_spn_conditional_scope_raises_not_implemented():
    data = _randn(20, 3)
    scope = Scope(query=[0, 1], evidence=[2])
    leaf = Normal(scope=Scope([0, 1]), out_channels=1)

    def partitioning_method(_):
        return torch.tensor([1, 1])

    def clustering_method(cluster_data):
        return torch.remainder(torch.arange(cluster_data.shape[0]), 2).to(cluster_data.device)

    with pytest.raises(NotImplementedError):
        learn_spn(
            data,
            leaf_modules=leaf,
            scope=scope,
            partitioning_method=partitioning_method,
            clustering_method=clustering_method,
            min_instances_slice=2,
            min_features_slice=2,
        )


def test_prune_sums_flattens_cat_of_cat_inputs():
    scope = Scope([0])
    left_a = Normal(scope=scope, out_channels=1)
    left_b = Normal(scope=scope, out_channels=1)
    right_a = Normal(scope=scope, out_channels=1)
    right_b = Normal(scope=scope, out_channels=1)

    child_sum1 = Sum(inputs=Cat([left_a, left_b], dim=2), out_channels=1)
    child_sum2 = Sum(inputs=Cat([right_a, right_b], dim=2), out_channels=1)
    root_sum = Sum(inputs=[child_sum1, child_sum2], out_channels=1)

    prune_sums(root_sum)

    assert isinstance(root_sum.inputs, Cat)
    assert len(root_sum.inputs.inputs) == 4


def test_prune_sums_descends_into_product_with_non_cat_input():
    scope = Scope([0])
    inner_sum = Sum(inputs=Normal(scope=scope, out_channels=1), out_channels=1)
    product = Product(inputs=inner_sum)

    prune_sums(product)

    assert isinstance(product.inputs, Sum)
