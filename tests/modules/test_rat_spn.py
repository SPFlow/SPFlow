import time

from torch.utils.data import DataLoader

from spflow.meta.data import Scope
from spflow.modules.rat.region_graph import random_region_graph
from spflow.modules.leaf import Normal
from spflow.modules.rat.rat_spn_new import RatSPN
from spflow.modules.factorize import Factorize
from collections import deque
from spflow.meta.data import Scope
import pytest
from spflow import log_likelihood, marginalize, sample
from spflow.meta.dispatch import init_default_sampling_context, SamplingContext
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
from spflow.modules.factorize import Factorize
from tests.utils.leaves import make_normal_leaf, make_normal_data
from spflow.learn import train_gradient_descent
from spflow.meta.dispatch import (
    SamplingContext,
    dispatch,
    init_default_dispatch_context,
)


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
    rat_spn = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=3,
        n_region_nodes=2,
        n_leaf_nodes=1,
        num_repetitions=3,
        depth=2
    )
    data = make_normal_data(out_features=7, num_samples=1)
    ll = log_likelihood(rat_spn, data)
    test = 5

    """
    fac = Factorize(inputs=[normal_layer])
    out_p = OuterProduct(inputs=[fac], num_splits=2)
    sum_layer = Sum(inputs=out_p, out_channels=4, num_repetitions=3)
    data = make_normal_data(out_features=7, num_samples=1)
    log_likelihood(sum_layer, data)


    region_graph = random_region_graph(scope, depth=3, replicas=1)
    rat_spn = RatSPN(
        region_graph,
        [normal_layer],
        n_root_nodes=1,
        n_region_nodes=1,
        n_leaf_nodes=1,
    )
    rat_list = rat_spn.factorize(2)

    test = 5
    """

def make_dataset(num_features_continuous, num_features_discrete, num_clusters, num_samples):
    # Collect data and data domains
    BINS = 100
    data = []
    domains = []

    # Construct continuous features
    for i in range(num_features_continuous):
        #domains.append(Domain.continuous_inf_support())
        feat_i = []

        # Create a multimodal feature
        #for j in range(num_clusters):
        #    feat_i.append(torch.randn(num_samples) + j * 3 * torch.rand(1) + 3 * j)
        for j in range(num_clusters):
            feat_i.append(torch.randn(num_samples) * 1.0 + j * 3 * torch.randint(low=1, high=10, size=(
            1,)) / 10 + 5 * j - num_clusters * 2.5)

        data.append(torch.cat(feat_i))


    data = torch.stack(data, dim=1)
    #data = data.view(data.shape[0], 1, num_features_continuous + num_features_discrete)
    data = data.view(data.shape[0], num_features_continuous + num_features_discrete, 1)
    data = data[torch.randperm(data.shape[0])]
    return data#, domains

def visualize(data, spn, samples):
    #samples = sample(spn, 10000)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    clip_min = -10  # Replace with desired minimum value
    clip_max = 10  # Replace with desired maximum value
    clipping_range = (clip_min, clip_max)

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):

        rng = clipping_range
        bins = 100

        # samples

        width_s = (samples[:, i].max() - samples[:, i].min()) / bins
        hist = torch.histogram(samples[:, i], bins=bins, density=True, range=rng)
        bin_edges = hist.bin_edges
        density = hist.hist

        # Center bars on value (e.g. bar for value 0 should have its center at value 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, density, width=width_s * 0.8, alpha=0.5, label="Samples")


        # data

        #width = (data[:, :, i].max() - data[:, :, i].min()) / bins
        width = (data[:, i].max() - data[:, i].min()) / bins
        #hist = torch.histogram(data[:, :, i], bins=bins, density=True, range=rng)
        hist = torch.histogram(data[:, i], bins=bins, density=True, range=rng)
        bin_edges = hist.bin_edges
        density = hist.hist
        

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, density, width=width * 0.8, alpha=0.5, label="Data")


        dummy = torch.full((bin_centers.shape[0], data.shape[1]), np.nan)
        dummy[:, i] = bin_centers
        with torch.no_grad():
            log_probs = spn(dummy)
        probs = log_probs[:,:,0].exp().squeeze(-1).numpy() # choose a channel and repetition
        ax.plot(bin_centers, probs, linewidth=2, label="Likelihood")


        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")

        ax.set_title(f"Feature {i} ")
        ax.legend()
    plt.tight_layout()
    plt.show()

def test_rat_spn_hist():
    # ToDo: MixtureLayer; repetition dimension for leaf layers; repitition dimension for elementwise product etc.

    # Scheinbar funktioniert es soweit nur mit einer repitition / Problem liegt bei repetition
    # Mögliches Problem: Weights sind nicht richtig für repitition -> nicht richtig normalisiert

    torch.manual_seed(0)
    num_features = 4
    out_channels = 10#10
    num_repetitions = 10
    n_samples = 10000

    data = make_dataset(num_features, 0, 3, n_samples).squeeze(-1)


    dataset = torch.utils.data.TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)


    random_variables = list(range(num_features))
    scope = Scope(random_variables)
    normal_layer = Normal(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)
    """
    epochs = 3
    batch_size = 128
    depth = 2
    num_sums = 20
    num_leaves = 10
    num_repetitions = 10
    lr = 0.01
    """
    start_time = time.time()
    rat_spn = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=1,
        n_region_nodes=10, #30
        num_repetitions=num_repetitions,#1,
        depth=2,
        outer_product=False,
    )

    print("Time to build SPN: ", time.time() - start_time)

    print("Number of parameters:", sum(p.numel() for p in normal_layer.parameters() if p.requires_grad))

    for name, ch in rat_spn.root_node.named_children():
        print("Number of parameters:", name, sum(p.numel() for p in ch.parameters() if p.requires_grad))

    train_gradient_descent(rat_spn, dataloader, lr=0.5, epochs=5, verbose=True)

    rat_spn.eval()
    dispatch_ctx = init_default_dispatch_context()
    #ll = log_likelihood(rat_spn, data, dispatch_ctx=dispatch_ctx)

    n_samples = data.shape[0]
    for i in range(10):
        sample_data = torch.full((n_samples, num_features), torch.nan)
        channel_index = torch.full((n_samples, rat_spn.root_node.out_features), fill_value=0)
        mask = torch.full((n_samples, rat_spn.root_node.out_features), True, dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_idx=i)
        samples = sample(rat_spn, sample_data, is_mpe=False, sampling_ctx=sampling_ctx, dispatch_ctx=dispatch_ctx)

        visualize(data, rat_spn, samples)

def test_sample():
    num_features = 4
    out_channels = 2
    num_repetitions = 3
    n_samples = 10
    random_variables = list(range(num_features))
    scope = Scope(random_variables)
    normal_layer = Normal(scope=scope, out_channels=out_channels, num_repetitions=num_repetitions)

    fac = Factorize(inputs=[normal_layer], depth=2, num_repetitions=3)
    data = make_dataset(num_features, 0, 3, 10000).squeeze(-1)

    visualize(data, fac)

    for i in range(fac.out_channels):
        data = torch.full((n_samples, num_features), torch.nan)
        channel_index = torch.full((n_samples, num_features), fill_value=i)
        mask = torch.full((n_samples, num_features), True, dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_idx=0)
        samples = sample(fac, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, fac.scope.query]
        assert torch.isfinite(samples_query).all()

def test_digits_sampling():

    # Parameters

    epochs = 50
    lr = 0.1
    batch_size = 128
    depth = 5
    n_region_nodes = 32
    num_leaves = 32
    num_repetitions = 10
    n_root_nodes = 1
    num_features = 64
    n_samples = 16

    random_variables = list(range(num_features))
    scope = Scope(random_variables)

    normal_layer = Normal(scope=scope, out_channels=num_leaves, num_repetitions=num_repetitions)

    rat_spn = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,  # 30
        num_repetitions=num_repetitions,  # 1,
        depth=depth,
        outer_product=False,
    )

    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size)
    #test_digits = next(iter(train_dataloader))[0][:n_samples]
    #visualize_digits(test_digits)
    #return
    optimizer = torch.optim.Adam(rat_spn.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1e-1, verbose=True)

    train_gradient_descent(rat_spn, train_dataloader, lr=lr, epochs=epochs, scheduler=lr_scheduler, verbose=True)

    sample_data = torch.full((n_samples, num_features), torch.nan)
    channel_index = torch.full((n_samples, rat_spn.root_node.out_features), fill_value=0)
    mask = torch.full((n_samples, rat_spn.root_node.out_features), True, dtype=torch.bool)
    sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask, repetition_idx=0)
    samples = sample(rat_spn, sample_data, is_mpe=False, sampling_ctx=sampling_ctx)
    visualize_digits(torch.clip(samples, min=0))

def test_digits_classification():
    epochs = 20
    lr = 0.1
    batch_size = 128
    depth = 5
    n_region_nodes = 32
    num_leaves = 32
    num_repetitions = 10
    n_root_nodes = 3
    num_features = 64
    n_samples = 16

    random_variables = list(range(num_features))
    scope = Scope(random_variables)

    normal_layer = Normal(scope=scope, out_channels=num_leaves, num_repetitions=num_repetitions)

    rat_spn = RatSPN(
        leaf_modules=[normal_layer],
        n_root_nodes=n_root_nodes,
        n_region_nodes=n_region_nodes,  # 30
        num_repetitions=num_repetitions,  # 1,
        depth=depth,
        outer_product=False,
    )
    train_dataloader, val_dataloader, test_dataloader = load_dataset(batch_size)

    optimizer = torch.optim.Adam(rat_spn.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1e-1, verbose=True)

    train_gradient_descent(rat_spn, train_dataloader, lr=lr, epochs=epochs, is_classification=True, scheduler=lr_scheduler, verbose=True, validation_dataloader=val_dataloader)

    test_accuracy = get_test_accuracy(rat_spn, test_dataloader)
    print(f"Test accuracy: {test_accuracy}")

def get_test_accuracy(model, dataloader):
    all_X = []
    all_y = []
    for batch in dataloader:
        X,y = batch
        all_X.append(X)
        all_y.append(y)

    X = torch.cat(all_X)
    y = torch.cat(all_y)
    ll = log_likelihood(model, X)
    y_pred = torch.argmax(ll.squeeze(1), dim=1)
    accuracy = (y_pred == y).float().mean().item()
    return accuracy


def visualize_digits(samples):
    images = samples.reshape(-1, 8, 8)
    images = images * 255.0
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # 4x4 grid for 16 samples
    fig.suptitle("Digit Samples")

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray', interpolation='nearest')  # Display each image
        ax.axis('off')  # Turn off axis for cleaner visualization

    plt.tight_layout()
    plt.show()


def load_dataset(batch_size):
    import torch
    from torch.utils.data import DataLoader, random_split, TensorDataset
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Load the dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Filter the data to include only the first three classes (0, 1, 2)
    mask = np.isin(y, [0,1,2])
    X = X[mask]
    y = y[mask]

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    #visualize_digits(X_tensor[:16])

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    data_train, data_val, data_test = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
