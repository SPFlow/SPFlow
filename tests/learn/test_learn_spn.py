from itertools import combinations, product
import pytest

import numpy as np
import torch

from spflow.learn import learn_spn
from spflow.learn.learn_spn import cluster_by_kmeans, prune_sums
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.module import Module
from spflow.modules.sums import Sum
from spflow.utils.rdc import rdc


def test_kmeans():
    # simulate cluster data
    # cluster = [torch.randn((100, 1))+ i*100.0 for i in range(num_cluster)]

    cluster_1 = torch.randn((100, 1)) - 20.0
    cluster_2 = torch.randn((100, 1)) - 10.0
    cluster_3 = torch.randn((100, 1))
    cluster_4 = torch.randn((100, 1)) + 10.0
    cluster_5 = torch.randn((100, 1)) + 20.0

    # compute clusters using k-means
    cluster_mask = cluster_by_kmeans(
        torch.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]).float(), n_clusters=5
    )

    assert len(torch.unique(cluster_mask)) == 5


def make_rdc_data(n_samples=1000):
    feature1 = torch.randn(n_samples)  # Normal distribution
    feature2 = torch.rand(n_samples) * 4 - 2  # Uniform distribution [-2, 2]
    feature3 = torch.distributions.Exponential(1.0).sample((n_samples,))  # Exponential distribution
    feature4 = torch.distributions.Binomial(10, 0.5).sample((n_samples,))  # Binomial distribution

    data = torch.stack((feature1, feature2, feature3, feature4), dim=1)
    return data


def test_rdc():
    from networkx import connected_components as ccnp, from_numpy_array

    # Generate synthetic data
    data = make_rdc_data()
    threshold = 0.3

    # Compute RDC
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

    for partition_id in torch.sort(torch.unique(partition_ids), dim=-1)[0]:  # uc
        partitions.append(torch.where(partition_ids == partition_id))  # uc

    assert len(partitions) == 4


@pytest.mark.parametrize(
    "leaf_channel,sum_channel",
    list(product([1, 2], [1, 2])),
)
def test_multiple_features(leaf_channel, sum_channel):
    # Create leaf layer with Gaussian distributions
    scope = Scope(list(range(5)))
    leaf_layer = Normal(scope=scope, out_channels=leaf_channel)

    # Learn SPN structure from data
    # Construct synthetic data for demonstration with five different clusters
    cluster_1 = torch.randn(200, 5) + torch.tensor([0, 0, 0, 0, 0])
    cluster_2 = torch.randn(200, 5) + torch.tensor([5, 5, 5, 5, 5])
    cluster_3 = torch.randn(200, 5) + torch.tensor([-5, -5, -5, -5, -5])
    cluster_4 = torch.randn(200, 5) + torch.tensor([10, 0, -10, 5, -5])
    cluster_5 = torch.randn(200, 5) + torch.tensor([-10, 5, 10, -5, 0])
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

    data = torch.randn(16, 1)
    lls_before = root_sum.log_likelihood(data)
    num_sums_before = sum(1 for m in root_sum.modules() if isinstance(m, Sum))

    prune_sums(root_sum)

    lls_after = root_sum.log_likelihood(data)
    num_sums_after = sum(1 for m in root_sum.modules() if isinstance(m, Sum))

    assert num_sums_after < num_sums_before
    assert torch.allclose(lls_before, lls_after)
