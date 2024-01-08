import itertools
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.modules.module import log_likelihood
from spflow.structure.spn import PartitionLayer, ProductNode, SumNode
from spflow.modules.node import Gaussian
from spflow.modules.node import toLayerBased
from spflow.modules.node import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_partition_layer_likelihood(do_for_all_backends):
    input_partitions = [
        [Gaussian(Scope([0])), Gaussian(Scope([0]))],
        [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
        [Gaussian(Scope([2]))],
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
    layer_based_spn = toLayerBased(layer_spn)

    dummy_data = tl.tensor([[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.25, 0.0, 1.0]])

    layer_ll = log_likelihood(layer_spn, dummy_data)
    nodes_ll = log_likelihood(nodes_spn, dummy_data)
    lb_ll = log_likelihood(layer_based_spn, dummy_data)

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))
    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(tl.tensor(lb_ll, dtype=tl.float64))))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    input_partitions = [
        [Gaussian(Scope([0])), Gaussian(Scope([0]))],
        [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
        [Gaussian(Scope([2]))],
    ]

    layer_spn = SumNode(
        children=[PartitionLayer(child_partitions=input_partitions)],
        weights=[0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
    )
    dummy_data = tl.tensor([[1.0, 0.25, 0.0], [0.0, 1.0, 0.25], [0.25, 0.0, 1.0]])

    layer_ll = log_likelihood(layer_spn, dummy_data)
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer_spn)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(layer_ll_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
