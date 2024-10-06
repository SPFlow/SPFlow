from spflow.meta.data import Scope
from spflow.modules.rat.region_graph import random_region_graph
from spflow.modules.leaf import Normal
from spflow.modules.rat.rat_spn import RatSPN
from collections import deque
from spflow.meta.data import Scope
import pytest
from spflow import log_likelihood, marginalize
from spflow.learn import train_gradient_descent
from spflow.modules import Sum, Product
from spflow.modules.base_product import BaseProduct
from spflow.modules.ops.cat import Cat
from tests.utils.leaves import make_normal_data
from spflow.learn.learn_spn import learn_spn
from spflow.learn.learn_spn import cluster_by_kmeans, partition_by_rdc
from scipy.stats import multivariate_normal
from spflow.learn.expectation_maximization import expectation_maximization
import torch
import random
from spflow.modules import Sum
from spflow.modules import OuterProduct
from spflow.modules import ElementwiseProduct

from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

def test_rat_spn():
    out_channels = 2
    random_variables = list(range(7))
    scope = Scope(random_variables)
    normal_layer = Normal(scope=scope, out_channels=out_channels)
    create_spn([normal_layer])
    region_graph = random_region_graph(scope, depth=3, replicas=1)
    rat_spn = RatSPN(
        region_graph,
        [normal_layer],
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )
    rat_list = list_modules_by_depth(rat_spn)

    test = 5

def create_spn(leaf_modules):

    def get_leaves(scope):
        leaves = []
        s = set(scope.query)
        for leaf_module in leaf_modules:
            leaf_scope = set(leaf_module.scope.query)
            scope_inter = s.intersection(leaf_scope)
            if len(scope_inter) > 0:
                leaf_layer = leaf_module.__class__(scope=Scope(sorted(scope_inter)),
                                                   out_channels=leaf_module.out_channels)

                leaves.append(leaf_layer)
        return leaves


    scope = leaf_modules[0].scope
    for leaf in leaf_modules[1:]:
        scope = scope.join(leaf.scope)

    shuffled_rvs = scope.query.copy()
    random.shuffle(shuffled_rvs)
    n_splits = 4
    split_rvs = np.array_split(shuffled_rvs, n_splits)
    leaves = []
    for split in split_rvs:
        s = Scope(sorted(split))
        leaves.append(get_leaves(s))

    input = []
    for leaf in leaves:
        for l in leaf:
            input.append(l)

    inputs = leaf_modules
    layer2 = Sum
    layer1 = OuterProduct(inputs=inputs)
    root_node = Sum(inputs=layer1, out_channels=1)

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
    region_graph = random_region_graph(scope, depth=10, replicas=10)
    rat_spn = RatSPN(
        region_graph,
        [normal_layer],
        n_root_nodes=10,
        n_region_nodes=10,
        n_leaf_nodes=10,
    )
    analyze_spn(rat_spn.root_node)
    heatmap(rat_spn.root_node, X, y)



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

    expectation_maximization(spn, torch.tensor(X, dtype=torch.float32), verbose=True)

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
        elif isinstance(spn, BaseProduct):
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
            level_modules.append(current_module)

            # Add children of the current module to the queue for the next level
            for child in current_module.children():
                queue.append(child)

        # Append the list of modules at the current depth level to the result
        result.append(level_modules)

    return result