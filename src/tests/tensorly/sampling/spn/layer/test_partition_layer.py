import itertools
import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import Gaussian
from spflow.tensorly.structure.spn import PartitionLayer, ProductNode, SumNode
from spflow.tensorly.structure.spn.node.sum_node import toLayerBased, updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_partition_layer_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    input_partitions = [
        [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
        ],
        [
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([1]), mean=-5.0, std=0.01),
            Gaussian(Scope([1]), mean=0.0, std=0.01),
        ],
        [Gaussian(Scope([2]), mean=10.0, std=0.01)],
    ]

    layer_spn = SumNode(
        children=[PartitionLayer(child_partitions=input_partitions)],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )

    nodes_spn = SumNode(
        children=[
            ProductNode(
                children=[
                    input_partitions[0][i],
                    input_partitions[1][j],
                    input_partitions[2][k],
                ]
            )
            for (i, j, k) in itertools.product([0, 1], [0, 1, 2], [0])
        ],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )

    expected_mean = (
        0.2 * tl.tensor([3.0, 1.0, 10.0])
        + 0.1 * tl.tensor([3.0, -5.0, 10.0])
        + 0.2 * tl.tensor([3.0, 0.0, 10.0])
        + 0.2 * tl.tensor([1.0, 1.0, 10.0])
        + 0.2 * tl.tensor([1.0, -5.0, 10.0])
        + 0.1 * tl.tensor([1.0, 0.0, 10.0])
    )

    layerbased_spn = toLayerBased(layer_spn)

    layer_samples = sample(layer_spn, 10000)
    nodes_samples = sample(nodes_spn, 10000)
    layerbased_samples = sample(layerbased_spn, 10000)

    tc.assertTrue(np.allclose(tl.mean(nodes_samples, axis=0), expected_mean, atol=0.01, rtol=0.1))
    tc.assertTrue(
        np.allclose(
            tl.mean(layer_samples, axis=0),
            tl.mean(nodes_samples, axis=0),
            atol=0.01,
            rtol=0.1,
        )
    )

    # sample from multiple outputs (with same scope)
    tc.assertRaises(
        ValueError,
        sample,
        list(layer_spn.children)[0],
        1,
        sampling_ctx=SamplingContext([0], [[0, 1]]),
    )

    tc.assertTrue(
        np.allclose(
            tl.mean(layer_samples, axis=0),
            tl.mean(layerbased_samples, axis=0),
            atol=0.01,
            rtol=0.1,
        )
    )

    # sample from multiple outputs (with same scope)
    tc.assertRaises(
        ValueError,
        sample,
        list(layerbased_spn.children)[0],
        1,
        sampling_ctx=SamplingContext([0], [[0, 1]]),
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    input_partitions = [
        [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
        ],
        [
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([1]), mean=-5.0, std=0.01),
            Gaussian(Scope([1]), mean=0.0, std=0.01),
        ],
        [Gaussian(Scope([2]), mean=10.0, std=0.01)],
    ]

    layer_spn = SumNode(
        children=[PartitionLayer(child_partitions=input_partitions)],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )

    layer_samples = sample(layer_spn, 10000)
    samples_mean = tl_toNumpy(layer_samples).mean()
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer_spn)
            layer_samples_updated = sample(layer_updated, 10000)
            samples_mean_updated = tl_toNumpy(layer_samples_updated).mean()
            tc.assertTrue(np.allclose(samples_mean, samples_mean_updated, atol=0.01, rtol=0.1))

def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    input_partitions = [
        [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
        ],
        [
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([1]), mean=-5.0, std=0.01),
            Gaussian(Scope([1]), mean=0.0, std=0.01),
        ],
        [Gaussian(Scope([2]), mean=10.0, std=0.01)],
    ]

    layer = SumNode(
        children=[PartitionLayer(child_partitions=input_partitions)],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )
    samples = sample(layer, 100)
    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)

    samples = sample(layer, 100)
    tc.assertTrue(samples.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    input_partitions = [
        [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
        ],
        [
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([1]), mean=-5.0, std=0.01),
            Gaussian(Scope([1]), mean=0.0, std=0.01),
        ],
        [Gaussian(Scope([2]), mean=10.0, std=0.01)],
    ]

    layer = SumNode(
        children=[PartitionLayer(child_partitions=input_partitions)],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )
    samples = sample(layer, 100)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)

    samples = sample(layer, 100)
    tc.assertTrue(samples.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
