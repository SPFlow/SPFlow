from spflow.modules.leaf import Normal, Bernoulli, Poisson
from tests.fixtures import auto_set_test_seed, auto_set_test_device
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
from spflow.learn.learn_spn import cluster_by_kmeans, partition_by_rdc, prune_sums
from scipy.stats import multivariate_normal
import torch
from collections import deque
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from spflow.utils.rdc import rdc
from networkx import connected_components as ccnp, from_numpy_array
from itertools import combinations
matplotlib.use('Agg')

out_features = 5
out_channels = 2

def test_kmeans():
    torch.manual_seed(0)
    # simulate cluster data
    #cluster = [torch.randn((100, 1))+ i*100.0 for i in range(num_cluster)]

    cluster_1 = torch.randn((100, 1)) - 20.0
    cluster_2 = torch.randn((100, 1)) - 10.0
    cluster_3 = torch.randn((100, 1))
    cluster_4 = torch.randn((100, 1)) + 10.0
    cluster_5 = torch.randn((100, 1)) + 20.0

    # compute clusters using k-means
    cluster_mask = cluster_by_kmeans(
        torch.tensor(torch.vstack([cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]), dtype=torch.float32), n_clusters=5
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


    # Generate synthetic data
    data = make_rdc_data()
    threshold = 0.3

    # Compute RDC
    rdcs = torch.eye(data.shape[1], device=data.device)
    for i, j in combinations(range(data.shape[1]), 2):
        r = rdc(data[:, i], data[:, j])
        rdcs[j][i] = rdcs[i][j] = r


    rdcs[rdcs < threshold] = 0.0
    adj_mat = rdcs

    partition_ids = torch.zeros(data.shape[1], dtype=torch.int)

    for i, c in enumerate((ccnp(from_numpy_array(np.array(adj_mat.cpu()))))):
        partition_ids[list(c)] = i + 1

    partition_ids.to(data.device)

    partitions = []

    for partition_id in torch.sort(torch.unique(partition_ids), axis=-1)[0]:  # uc
        partitions.append(torch.where(partition_ids == partition_id))  # uc

    assert len(partitions) == 4

def test_make_moons():
    visualize = False
    torch.manual_seed(0)
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
    assert num_params_after_pruning < num_params

    if visualize:
        # Visualize the contours of the learned SPN
        heatmap(spn, X, y)


def plot_contours(mean, std):
    x, y = np.mgrid[-10:10:.05, -10:10:.05]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, np.diag(std ** 2))
    plt.contour(x, y, rv.pdf(pos), levels=5, colors='black')

def heatmap(spn, X, y):
    torch.cuda.empty_cache()
    torch.set_default_device("cpu")
    device = torch.device("cpu")
    spn.to(device)
    # Create a meshgrid of points over the feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten the grid so that you can pass it through the SPN
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Assuming you have a trained SPN called `spn`
    # Calculate the likelihoods (probabilities) for each point in the grid
    probs = log_likelihood(spn, torch.tensor(grid, dtype=torch.float32))

    # Reshape the probabilities back into a grid form
    probs = probs[:,0,0].reshape(xx.shape).detach().numpy()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))

    plt.contour(xx, yy, np.exp(probs), levels=10, cmap="viridis")

    # Optionally, overlay the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="w", s=50, alpha=0.8)

    plt.title("SPN Probability Heatmap")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


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

def list_modules_by_depth(root):
    if not root:
        return []

    # Initialize the queue with the root node, the list to store the result
    result = []
    queue = deque([root])

    while queue:
        # Start processing a new depth level
        level_modules = []
        level_size = len(queue)  # The number of nodes at this depth

        for _ in range(level_size):
            current_module = queue.popleft()
            if not(current_module.__class__.__name__ == "Cat" or current_module.__class__.__name__ == "ModuleList"):
                level_modules.append(current_module.__class__.__name__)

            # Add children of the current module to the queue for the next level
            for child in current_module.children():
                queue.append(child)

        # Append the list of modules at the current depth level to the result
        if level_modules:
            result.append(level_modules)

    return result








