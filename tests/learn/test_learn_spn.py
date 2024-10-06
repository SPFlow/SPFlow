from spflow.modules.leaf import Normal, Bernoulli, Poisson
from tests.fixtures import auto_set_test_seed
import unittest

from itertools import product

from spflow.meta.data import Scope
import pytest
from spflow import log_likelihood, marginalize
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, Product
from spflow.modules.ops.cat import Cat
from tests.utils.leaves import make_normal_data
from spflow.learn.learn_spn import learn_spn
from spflow.learn.learn_spn import cluster_by_kmeans, partition_by_rdc
from scipy.stats import multivariate_normal
import torch

from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

out_features = 5
out_channels = 2

def clustering_fn(x):
    # split into two approximately equal sized clusters
    mask = torch.zeros(x.shape[0])
    mask[int(x.shape[0] / 2) :] = 1
    return mask


def partitioning_fn(x):
    ids = torch.zeros(x.shape[1])

    if not partitioning_fn.alternate or partitioning_fn.partition:
        # split into two approximately equal sized partitions
        partitioning_fn.partition = False
        ids[: int(x.shape[1] / 2)] = 1
    else:
        partitioning_fn.partition = True
    return ids

def test_make_blobs():
    torch.manual_seed(0)

    X, y = make_blobs(n_samples=1000,centers=2, n_features=4, random_state=42)

    scope = Scope(list(range(4)))
    normal_layer = Normal(scope=scope, out_channels=1)

    spn = learn_spn(
        torch.tensor(X, dtype=torch.float32),
        leaf_modules=normal_layer,
        fit_params=False,
        min_instances_slice=2, #51
    )

    heatmap(spn, X, y)


@pytest.mark.parametrize("num_cluster", [1, 2, 3, 4, 5])
def test_kmeans(num_cluster):
    torch.manual_seed(0)

    # simulate cluster data
    #cluster = [torch.randn((100, 1))+ i*100.0 for i in range(num_cluster)]

    cluster_1 = torch.randn((100, 1)) - 20.0
    cluster_2 = torch.randn((100, 1)) - 10.0
    cluster_3 = torch.randn((100, 1))
    cluster_4 = torch.randn((100, 1)) + 10.0
    cluster_5 = torch.randn((100, 1)) + 20.0

    """
    cluster_1 = torch.randn((100, 1))
    cluster_2 = torch.randn((100, 1)) + 10.0
    cluster_3 = torch.randn((100, 1)) + 20.0
    cluster_4 = torch.randn((100, 1)) + 30.0
    cluster_5 = torch.randn((100, 1)) + 40.0
    """

    # compute clusters using k-means
    cluster_mask = cluster_by_kmeans(
        torch.tensor(torch.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]), dtype=torch.float32), n_clusters=5
        #torch.vstack(cluster), n_clusters=num_cluster
    )

    cluster_ids = range(num_cluster)

    for i in range(num_cluster):
        assert(torch.all(cluster_mask[100*i:100*(i+1)] == i))
    """
    # cluster id can either be 0,1 or 2
    cluster_id = cluster_mask[0]
    cluster_ids.remove(cluster_id)
    
    # make sure all first 100 entries have the same cluster i d
    assert(torch.all(cluster_mask[:100] == cluster_id))

    # second cluster id should be different from first
    cluster_id = cluster_mask[100]
    assert(cluster_id in cluster_ids)
    cluster_ids.remove(cluster_id)

    assert(torch.all(cluster_mask[100:200] == cluster_id))

    # third cluster id should be different from first two
    cluster_id = cluster_mask[200]
    assert(cluster_id in cluster_ids)
    cluster_ids.remove(cluster_id)

    assert(torch.all(cluster_mask[200:] == cluster_id))
    
    """

def test_rdc_partitioning():
    # set seed
    torch.manual_seed(0)

    # simulate partition data
    data_partition_1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.tensor([[1, 0.5], [0.5, 1]])).sample((100,))
    data_partition_2 = torch.randn((100, 1)) + 10.0

    # compute clusters using k-means
    partition_mask = partition_by_rdc(
        torch.hstack([data_partition_1, data_partition_2]),
        threshold=0.5,
    )

    # should be two partitions
    assert(len(torch.unique(partition_mask)) == 2)

    # check if partitions are correct (order is irrelevant)
    partition_1 = torch.where(partition_mask == 0)[0]
    assert(torch.all(partition_1 == torch.tensor([0, 1])) or torch.all(partition_1 == torch.tensor([2])))

def test_learn_1():

    X, y = make_blobs(n_samples=100,  n_features=out_features, random_state=0)
    data = torch.tensor(X, dtype=torch.float32)
    scope = Scope(list(range(out_features)))


    normal_layer = Normal(scope= scope, out_channels=out_channels)

    # ----- min_features_slice > scope size (no splitting or clustering) -----

    #partitioning_fn.alternate = True
    #partitioning_fn.partition = True

    spn = learn_spn(
        data,
        leaf_modules=normal_layer,
        fit_params=False,
        min_features_slice=2,
    )

def test_learn_spn_2():
    # set seed
    torch.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)

    data = make_normal_data(num_samples=100, out_features=3)
    scope = Scope(list(range(3)))
    normal_layer = Normal(scope=scope, out_channels=out_channels)

    partitioning_fn.alternate = True
    partitioning_fn.partition = True

    spn = learn_spn(
        data,
        leaf_modules=normal_layer,
        partitioning_method=partitioning_fn,
        clustering_method=clustering_fn,
        fit_params=False,
        min_instances_slice=51,
    )

def test_learn_spn_3():
    # set seed
    torch.manual_seed(0)
    # np.random.seed(0)
    # random.seed(0)

    data = make_normal_data(num_samples=100, out_features=9)
    scope = Scope(list(range(3)))
    normal_layer = Normal(scope=scope, out_channels=out_channels)
    scope_ber = Scope(list(range(3,6)))
    bernoulli_layer = Bernoulli(scope=scope_ber, out_channels=out_channels)
    scope_p = Scope(list(range(6,9)))
    poisson_layer = Poisson(scope=scope_p, out_channels=out_channels)
    layers = [normal_layer, bernoulli_layer, poisson_layer]

    partitioning_fn.alternate = True
    partitioning_fn.partition = True

    spn = learn_spn(
        data,
        leaf_modules=layers,
        partitioning_method=partitioning_fn,
        clustering_method=clustering_fn,
        fit_params=False,
        min_instances_slice=51,
    )

    normal_samples = torch.randn(100, 3)

    bernoulli_samples = torch.bernoulli(torch.full((100, 3), 0.5))

    poisson_samples = torch.poisson(torch.full((100, 3), 2.0))

    data = torch.cat([normal_samples, bernoulli_samples, poisson_samples], dim=1)
    lls = log_likelihood(spn, data)

def test_make_moons():
    torch.manual_seed(3)
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42) #, random_state=42

    """
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.title("make_moons dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
    """

    scope = Scope(list(range(2)))
    normal_layer = Normal(scope=scope, out_channels=4)

    #partitioning_fn.alternate = True
    #partitioning_fn.partition = True
    spn = learn_spn(
        torch.tensor(X, dtype=torch.float32),
        leaf_modules=normal_layer,
        out_channels=3,
        #partitioning_method=partitioning_fn,
        #clustering_method=clustering_fn,
        min_instances_slice=70, #51
    )
    #analyze_spn(spn)
    heatmap(spn, X, y)
    means = [child.distribution.mean.detach().numpy()[:,0] for child in analyze_spn(spn)]
    stds = [child.distribution.std.detach().numpy()[:,0] for child in analyze_spn(spn)]
    test = 5

def plot_contours(mean, std):
    x, y = np.mgrid[-10:10:.05, -10:10:.05]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, np.diag(std ** 2))
    plt.contour(x, y, rv.pdf(pos), levels=5, colors='black')

def heatmap(spn, X, y):

    # Create a meshgrid of points over the feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten the grid so that you can pass it through the SPN
    grid = np.c_[xx.ravel(), yy.ravel()]

    #X, y = make_moons(n_samples=1000, noise=0.1, random_state=42) #, random_state=42

    #X_tensor = torch.tensor(X, dtype=torch.float32)
    #y_tensor = torch.tensor(y, dtype=torch.long)
    #moon_dataset = TensorDataset(X_tensor)

    #expectation_maximization(spn, torch.tensor(X, dtype=torch.float32), verbose=True)

    #dataloader = DataLoader(moon_dataset, batch_size=128, shuffle=True)

    #train_gradient_descent(spn, dataloader, lr=0.01, epochs=50, verbose=True)
    # Assuming you have a trained SPN called `spn`
    # Calculate the likelihoods (probabilities) for each point in the grid
    probs = log_likelihood(spn, torch.tensor(grid, dtype=torch.float32))

    # Reshape the probabilities back into a grid form
    probs = probs[:,0,0].reshape(xx.shape).detach().numpy()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    #plt.contourf(xx, yy, probs, levels=100, cmap="hot", alpha=0.8)
    #plt.colorbar(label="Probability")
    plt.contour(xx, yy, np.exp(probs), levels=10, cmap="viridis")

    # Optionally, overlay the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="w", s=50, alpha=0.8)

    plt.title("SPN Probability Heatmap")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    """
    n_samples = 100
    out_features = 2

    data = torch.full((n_samples, out_features), torch.nan)
    channel_index = torch.full((n_samples, out_features), fill_value=0)
    mask = torch.full((n_samples, out_features), True, dtype=torch.bool)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
    samples = sample(spn, data, sampling_ctx=sampling_ctx)

    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], color='red', label='Samples')
    plt.title("Samples from the SPN")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
    """


def analyze_spn(spn):
    counts = {'Sum': 0, 'Product': 0, 'Cat': 0, 'Leaf': 0}
    leaves = []
    def iterate_spn(spn):
        if isinstance(spn, Sum):
            counts['Sum'] += 1
            iterate_spn(spn.inputs)
        elif isinstance(spn, Product):
            counts['Product'] += 1
            iterate_spn(spn.inputs)
        elif isinstance(spn, Cat):
            counts['Cat'] += 1
            for i in spn.inputs:
                iterate_spn(i)
        else:
            leaves.append(spn)
            counts['Leaf'] += 1
            return
    iterate_spn(spn)
    print(counts)
    return leaves









