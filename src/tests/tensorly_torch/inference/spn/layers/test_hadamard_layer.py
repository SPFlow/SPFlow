import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import HadamardLayer, ProductNode, SumNode
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn.nodes.sum_node import toLayerBased
from spflow.tensorly.structure.spn.nodes.sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_hadamard_layer_likelihood(self):

        input_partitions = [
            [Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
            [Gaussian(Scope([3])), Gaussian(Scope([3])), Gaussian(Scope([3]))],
        ]

        layer_spn = SumNode(
            children=[HadamardLayer(child_partitions=input_partitions)],
            weights=[0.3, 0.2, 0.5],
        )

        nodes_spn = SumNode(
            children=[
                ProductNode(
                    children=[
                        input_partitions[0][i],
                        input_partitions[1][j],
                        input_partitions[2][k],
                        input_partitions[3][l],
                    ]
                )
                for (i, j, k, l) in [[0, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 2]]
            ],
            weights=[0.3, 0.2, 0.5],
        )
        layer_based_spn = toLayerBased(layer_spn)
        dummy_data = torch.tensor(
            [
                [1.0, 0.25, 0.0, -0.7],
                [0.0, 1.0, 0.25, 0.12],
                [0.25, 0.0, 1.0, 0.0],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        nodes_ll = log_likelihood(nodes_spn, dummy_data)
        lb_ll = log_likelihood(layer_based_spn, dummy_data)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))
        self.assertTrue(torch.allclose(layer_ll, torch.tensor(lb_ll, dtype=float)))

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        input_partitions = [
            [Gaussian(Scope([0]))],
            [Gaussian(Scope([1])), Gaussian(Scope([1])), Gaussian(Scope([1]))],
            [Gaussian(Scope([2]))],
            [Gaussian(Scope([3])), Gaussian(Scope([3])), Gaussian(Scope([3]))],
        ]

        layer_spn = SumNode(
            children=[HadamardLayer(child_partitions=input_partitions)],
            weights=[0.3, 0.2, 0.5],
        )
        dummy_data = tl.tensor(
            [
                [1.0, 0.25, 0.0, -0.7],
                [0.0, 1.0, 0.25, 0.12],
                [0.25, 0.0, 1.0, 0.0],
            ]
        )

        layer_ll = log_likelihood(layer_spn, dummy_data)
        for backend in backends:
            tl.set_backend(backend)
            layer_updated = updateBackend(layer_spn)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            self.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(layer_ll_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
