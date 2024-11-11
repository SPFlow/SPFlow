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
        for j in range(num_clusters):
            feat_i.append(torch.randn(num_samples) + j * 3 * torch.rand(1) + 3 * j)

        data.append(torch.cat(feat_i))


    data = torch.stack(data, dim=1)
    #data = data.view(data.shape[0], 1, num_features_continuous + num_features_discrete)
    data = data.view(data.shape[0], num_features_continuous + num_features_discrete, 1)
    data = data[torch.randperm(data.shape[0])]
    return data#, domains

def visualize(data, spn):
    #samples = sample(spn, 10000)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):

        rng = None
        bins = 100

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
        probs = log_probs[:,:,0].exp().squeeze(-1).numpy() # choose a channel
        ax.plot(bin_centers, probs, linewidth=2, label="Likelihood")


        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")

        ax.set_title(f"Feature {i} ")
        ax.legend()
    plt.tight_layout()
    plt.show()

def test_rat_spn_hist():

    # Scheinbar funktioniert es soweit nur mit einer repitition / Problem liegt bei repitition
    # Mögliches Problem: Weights sind nicht richtig für repitition -> nicht richtig normalisiert

    torch.manual_seed(0)
    num_features = 4
    out_channels = 10 #30

    data = make_dataset(num_features, 0, 6, 1000).squeeze(-1)


    dataset = torch.utils.data.TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    random_variables = list(range(num_features))
    scope = Scope(random_variables)
    normal_layer = Normal(scope=scope, out_channels=out_channels)
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
        n_region_nodes=20, #30
        n_leaf_nodes=1, # ToDo: Drop this parameter or override out_channelsin leaf_modules?
        num_repetitions=10,
        depth=2
    )
    #samples = sample(rat_spn, 10000)

    print("Time to build SPN: ", time.time() - start_time)

    print("Number of parameters:", sum(p.numel() for p in rat_spn.parameters() if p.requires_grad))

    train_gradient_descent(rat_spn, dataloader, lr=0.3, epochs=10, verbose=True) #0.2

    rat_spn.eval()

    visualize(data, rat_spn)

def test_sample():
    num_features = 7
    out_channels = 2
    n_samples = 10
    random_variables = list(range(num_features))
    scope = Scope(random_variables)
    normal_layer = Normal(scope=scope, out_channels=out_channels)

    fac = Factorize(inputs=[normal_layer], depth=2, num_repetitions=3)

    for i in range(fac.out_channels):
        data = torch.full((n_samples, num_features), torch.nan)
        channel_index = torch.full((n_samples, num_features), fill_value=i)
        mask = torch.full((n_samples, num_features), True, dtype=torch.bool)
        sampling_ctx = SamplingContext(channel_index=channel_index, mask=mask)
        samples = sample(fac, data, sampling_ctx=sampling_ctx)
        assert samples.shape == data.shape
        samples_query = samples[:, fac.scope.query]
        assert torch.isfinite(samples_query).all()
