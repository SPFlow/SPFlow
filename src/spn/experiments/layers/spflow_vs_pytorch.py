import sys

import os

#from spn.experiments.layers.Vectorized import to_layers, eval_layers, get_torch_spn
from glob import glob

from spn.experiments.layers.Vectorized import to_layers, get_torch_spn
from spn.experiments.layers.speedtest import eval_layers2, to_sparse

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spn
import tensorflow as tf
import torch
from matplotlib.font_manager import FontProperties
from spn.algorithms.Inference import log_likelihood
from spn.gpu.TensorFlow import eval_tf, eval_tf_graph, spn_to_tf_graph
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Gaussian
from torch import nn
from tqdm import tqdm

from spn.algorithms.layerwise import layers
from spn.algorithms.layerwise.distributions import Normal


def create_spflow_spn(n_feats):
    gaussians1 = []
    gaussians2 = []
    for i in range(n_feats):
        g1 = Gaussian(np.random.randn(), np.random.rand(), scope=i)
        g2 = Gaussian(np.random.randn(), np.random.rand(), scope=i)
        gaussians1.append(g1)
        gaussians2.append(g2)

    prods1 = []
    prods2 = []
    for i in range(0, n_feats, 2):
        p1 = Product([gaussians1[i], gaussians1[i + 1]])
        p2 = Product([gaussians2[i], gaussians2[i + 1]])
        prods1.append(p1)
        prods2.append(p2)

    sums = []
    for i in range(n_feats // 2):
        s = Sum(weights=[0.5, 0.5], children=[prods1[i], prods2[i]])
        sums.append(s)

    spflow_spn = Product(sums)
    assign_ids(spflow_spn)
    rebuild_scopes_bottom_up(spflow_spn)
    return spflow_spn


def create_pytorch_spn(n_feats):
    # Create SPN layers
    gauss = Normal(multiplicity=2, in_features=n_feats, in_channels=1)
    prod1 = layers.Product(in_features=n_feats, cardinality=2)
    sum1 = layers.Sum(in_features=n_feats / 2, in_channels=2, out_channels=1)
    prod2 = layers.Product(in_features=n_feats / 2, cardinality=n_feats // 2)

    # Stack SPN layers
    device = torch.device("cuda:0")
    pytorch_spn = nn.Sequential(gauss, prod1, sum1, prod2).to(device)
    return pytorch_spn


def run_pytorch(pytorch_spn, n_feats, batch_size, repetitions):
    pytorch_spn.eval()
    with torch.no_grad():
        print("Running PyTorch with: nfeat=%s, batch=%s" % (n_feats, batch_size))
        device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

        x = torch.rand(batch_size, n_feats, 1).to(device)
        # Warmup caches
        for i in range(10):
            ll = pytorch_spn(x)

        # Benchmark loop
        t = 0.0
        for i in tqdm(range(repetitions), desc="Repetition loop"):
            x = torch.rand(batch_size, n_feats, 1).to(device)
            t0 = time()
            ll = pytorch_spn(x)
            t += time() - t0

        pytorch_time = t / repetitions
        return pytorch_time


def run_spflow(spflow_spn, n_feats, batch_size, repetitions):
    print("Running SPFlow with: nfeat=%s, batch=%s" % (n_feats, batch_size))
    x = np.random.rand(batch_size, n_feats).astype(np.float32)

    # warmup
    for i in range(10):
        ll = log_likelihood(spflow_spn, x)

    # Run SPFlow spn
    t = 0.0
    for i in tqdm(range(repetitions), desc="Repetition loop"):
        x = np.random.rand(batch_size, n_feats).astype(np.float32)
        t0 = time()
        ll = log_likelihood(spflow_spn, x)
        t += time() - t0

    spflow_time = t / repetitions
    return spflow_time

def run_spflow_layers(layers, n_feats, batch_size, repetitions):
    print("Running SPFlow layers with: nfeat=%s, batch=%s" % (n_feats, batch_size))
    x = np.random.rand(batch_size, n_feats).astype(np.float32)

    # warmup
    for i in range(10):
        ll = eval_layers2(layers, x)

    # Run SPFlow spn
    t = 0.0
    for i in tqdm(range(repetitions), desc="Repetition loop"):
        x = np.random.rand(batch_size, n_feats).astype(np.float32)
        t0 = time()
        ll = eval_layers2(layers, x)
        t += time() - t0

    spflow_time = t / repetitions
    return spflow_time

def run_spflow_layers_torch(spn, n_feats, batch_size, repetitions):
    print("Running SPFlow layers torch with: nfeat=%s, batch=%s" % (n_feats, batch_size))
    device = "cuda"
    x = torch.rand(batch_size, n_feats, 1).to(device)

    # warmup
    for i in range(10):
        ll = spn(x)

    # Run SPFlow spn
    t = 0.0
    for i in tqdm(range(repetitions), desc="Repetition loop"):
        x = torch.rand(batch_size, n_feats, 1).to(device)
        t0 = time()
        ll = spn(x)
        t += time() - t0

    spflow_time = t / repetitions
    return spflow_time

def run_tf(spflow_spn, n_feats, batch_size, repetitions):
    print("Running TF with: nfeat=%s, batch=%s" % (n_feats, batch_size))
    x = np.random.rand(batch_size, n_feats).astype(np.float32)
    tf_graph, placeholder, _ = spn_to_tf_graph(spflow_spn, x, dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # warmup:
        for i in range(10):
            result = sess.run(tf_graph, feed_dict={placeholder: x})

        t = 0.0
        for i in tqdm(range(repetitions), desc="Repetition loop"):
            x = np.random.rand(batch_size, n_feats).astype(np.float32)
            t0 = time()
            result = sess.run(tf_graph, feed_dict={placeholder: x})
            t += time() - t0
    tf_time = t / repetitions
    return tf_time


def get_times(lib, n_feats, batch_size, repetitions):
    if lib == "pytorch":
        pytorch_spn = create_pytorch_spn(n_feats)
        return run_pytorch(pytorch_spn, n_feats, batch_size, repetitions)
    elif lib == "spflow":
        spflow_spn = create_spflow_spn(n_feats)
        return run_spflow(spflow_spn, n_feats, batch_size, repetitions)
    elif lib == "spflow-layer":
        spflow_spn = create_spflow_spn(n_feats)
        layers = to_layers(spflow_spn)
        to_sparse(layers)
        return run_spflow_layers(layers, n_feats, batch_size, repetitions)
    elif lib == "spflow-layer-pytorch":
        spflow_spn = create_spflow_spn(n_feats)
        layers = to_layers(spflow_spn)
        spn = get_torch_spn(layers).to('cuda')
        return run_spflow_layers_torch(spn, n_feats, batch_size, repetitions)
    elif lib == "spflow-tf":
        spflow_spn = create_spflow_spn(n_feats)
        return run_tf(spflow_spn, n_feats, batch_size, repetitions)
    else:
        raise Exception("Invalid lib")


def visualize():
    def load(lib):
        batch = np.loadtxt("results-batch-size-%s.csv" % lib, delimiter=",")
        nfeat = np.loadtxt("results-nfeat-%s.csv" % lib, delimiter=",")
        return batch, nfeat



    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.despine()
    # sns.set_context("poster")
    fig.set_figheight(6)
    fig.set_figwidth(9)

    names = [f[f.index('size-') + 5:-4] for f in glob('*size*.csv')]

    for name in names:
        batch_res, nfeat_res = load(name)
        axes[0, 0].plot(batch_res[:, 0], batch_res[:, 1], label=name)
        axes[0, 1].plot(nfeat_res[:, 0], nfeat_res[:, 1], label=name)
    # pytorch_batch, pytorch_nfeat = load("pytorch")
    # spflow_batch, spflow_nfeat = load("spflow")
    # tf_batch, tf_nfeat = load("spflow-tf")
    # spflow_layers_batch, spflow_layers_nfeat = load("spflow-layer")

    # Plot absolute values
    #axes[0, 0].plot(spflow_batch[:, 0], spflow_batch[:, 1], label="SPFlow")
    #axes[0, 0].plot(tf_batch[:, 0], tf_batch[:, 1], label="SPFlow-TF")
    #axes[0, 0].plot(pytorch_batch[:, 0], pytorch_batch[:, 1], label="Layerwise PyTorch")
    #axes[0, 0].plot(spflow_layers_batch[:, 0], spflow_layers_batch[:, 1], label="SPFlow Layers")

    #axes[0, 1].plot(spflow_nfeat[:, 0], spflow_nfeat[:, 1], label="SPFlow")
    #axes[0, 1].plot(tf_nfeat[:, 0], tf_nfeat[:, 1], label="SPFlow-TF")
    #axes[0, 1].plot(pytorch_nfeat[:, 0], pytorch_nfeat[:, 1], label="Layerwise PyTorch")
    #axes[0, 1].plot(spflow_layers_nfeat[:, 0], spflow_layers_nfeat[:, 1], label="SPFlow Layers")

    fontP = FontProperties()
    fontP.set_size("small")
    axes[0, 1].legend(loc="upper left", fancybox=True, framealpha=0.5, prop=fontP)

    axes[0, 0].set_xscale("log", basex=2)
    axes[0, 0].set_yscale("log", basey=10)
    axes[0, 1].set_xscale("log", basex=2)
    axes[0, 1].set_yscale("log", basey=10)

    # Plot relative improvements
    # axes[1, 0].plot(
    #     spflow_batch[:, 0],
    #     spflow_batch[:, 1] / pytorch_batch[:, 1],
    #     label="SPFlow/Layerwise PyTorch",
    # )
    # axes[1, 0].plot(
    #     tf_batch[:, 0],
    #     tf_batch[:, 1] / pytorch_batch[:, 1],
    #     label="SPFlow-TF/Layerwise PyTorch",
    # )
    # axes[1, 1].plot(
    #     spflow_nfeat[:, 0],
    #     spflow_nfeat[:, 1] / pytorch_nfeat[:, 1],
    #     label="SPFlow/Layerwise PyTorch",
    # )
    # axes[1, 1].plot(
    #     tf_nfeat[:, 0],
    #     tf_nfeat[:, 1] / pytorch_nfeat[:, 1],
    #     label="SPFlow-TF/Layerwise PyTorch",
    # )

    axes[1, 0].set_xscale("log", basex=2)
    # axes[1, 0].set_yscale("log", basey=10)
    axes[1, 1].set_xscale("log", basex=2)
    # axes[1, 1].set_yscale("log", basey=10)
    axes[0, 0].set_ylabel("Avg Time (s) over 100 Runs")
    axes[0, 1].set_ylabel("Avg Time (s) over 100 Runs")
    axes[1, 0].set_ylabel(r"Relative Time $\frac{t_x}{t_{PyTorch}}$")
    axes[1, 1].set_ylabel(r"Relative Time $\frac{t_x}{t_{PyTorch}}$")
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 1].set_xlabel("Features")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 1].set_xlabel("Features")
    axes[1, 1].legend(loc="upper left", fancybox=True, framealpha=0.5, prop=fontP)

    # Titles
    title = "vs".join(names) + " SPN Forward" #"SPFlow vs SPFlow-TF vs PyTorch: SPN Forward Pass"
    fig.suptitle(title)
    plt.savefig("benchmark.png", dpi=300)  # , bbox_inches="tight")


def benchmark(lib, max_i_batch, max_i_feat):
    # Set seed for reproducability
    np.random.seed(0)
    torch.manual_seed(0)

    repetitions = 100
    results = []
    default_nfeat = 1024
    default_batchsize = 1024

    list_n_feats = [2 ** i for i in range(2, max_i_feat)]

    tmp_n_feats = []
    for n_feats in tqdm(list_n_feats, desc="%s, Features" % lib):
        results.append(get_times(lib, n_feats, default_batchsize, repetitions))
        tmp_n_feats.append(n_feats)

        res_nfeats = np.c_[tmp_n_feats, np.array(results)]
        np.savetxt("results-nfeat-%s.csv" % lib, res_nfeats, delimiter=",")

    results = []
    list_batch_size = [2 ** i for i in range(2, max_i_batch)]
    tmp_batch_size = []
    for batch_size in tqdm(list_batch_size, desc="%s, Batch Size" % lib):
        results.append(get_times(lib, default_nfeat, batch_size, repetitions))
        tmp_batch_size.append(batch_size)

        res_batch = np.c_[tmp_batch_size, np.array(results)]
        np.savetxt("results-batch-size-%s.csv" % lib, res_batch, delimiter=",")


if __name__ == "__main__":
    #if sys.argv[1] == "benchmark":
        #lib = 'spflow-layer'  # sys.argv[2]
        #lib = 'spflow-layer-pytorch'
        lib = 'pytorch'
        max_i_batch = 10  # int(sys.argv[3])
        max_i_feat = 15  # int(sys.argv[4])
        benchmark(lib, max_i_batch, max_i_feat)
    #elif sys.argv[1] == "visualize":
        visualize()
