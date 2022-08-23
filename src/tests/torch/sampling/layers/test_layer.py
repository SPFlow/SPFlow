from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.sampling.nodes.node import sample
from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer, SPNHadamardLayer
from spflow.torch.inference.layers.layer import log_likelihood
from spflow.torch.sampling.layers.layer import sample
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.nodes.leaves.parametric.gaussian import log_likelihood
from spflow.torch.sampling.nodes.leaves.parametric.gaussian import sample
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import unittest
import itertools


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_sum_layer_sampling(self):

        input_nodes = [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
            Gaussian(Scope([0]), mean=0.0, std=0.01)
        ]

        layer_spn = SPNSumNode(children=[
            SPNSumLayer(n_nodes=3,
                children=input_nodes,
                weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]]),
            ],
            weights = [0.3, 0.4, 0.3]
        )

        nodes_spn = SPNSumNode(children=[
                SPNSumNode(children=input_nodes, weights=[0.8, 0.1, 0.1]),
                SPNSumNode(children=input_nodes, weights=[0.2, 0.3, 0.5]),
                SPNSumNode(children=input_nodes, weights=[0.2, 0.7, 0.1]),
            ],
            weights = [0.3, 0.4, 0.3]
        )

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        expected_mean = 0.3 * (0.8 * 3.0 + 0.1 * 1.0 + 0.1 * 0.0) + 0.4 * (0.2 * 3.0 + 0.3 * 1.0 + 0.5 * 0.0) + 0.3 * (0.2 * 3.0 + 0.7 * 1.0 + 0.1 * 0.0)
        self.assertTrue(torch.allclose(nodes_samples.mean(dim=0), torch.tensor([expected_mean]), atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(layer_samples.mean(dim=0), nodes_samples.mean(dim=0), atol=0.01, rtol=0.1))

        # sample from multiple outputs (with same scope)
        self.assertRaises(ValueError, sample, list(layer_spn.children())[0], 1, sampling_ctx=SamplingContext([0], [[0,1]]))

    def test_product_layer_sampling(self):

        input_nodes = [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([1]), mean=1.0, std=0.01),
            Gaussian(Scope([2]), mean=0.0, std=0.01)
        ]

        layer_spn = SPNSumNode(children=[
            SPNProductLayer(n_nodes=3, children=input_nodes)
            ],
            weights = [0.3, 0.4, 0.3]
        )

        nodes_spn = SPNSumNode(children=[
                SPNProductNode(children=input_nodes),
                SPNProductNode(children=input_nodes),
                SPNProductNode(children=input_nodes),
            ],
            weights = [0.3, 0.4, 0.3]
        )

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        self.assertTrue(torch.allclose(nodes_samples.mean(axis=0), torch.tensor([3.0, 1.0, 0.0]), atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))

        # sample from multiple outputs (with same scope)
        self.assertRaises(ValueError, sample, list(layer_spn.children())[0], 1, sampling_ctx=SamplingContext([0], [[0,1]]))

    def test_partition_layer_sampling(self):
        
        input_partitions = [
            [Gaussian(Scope([0]), mean=3.0, std=0.01), Gaussian(Scope([0]), mean=1.0, std=0.01)],
            [Gaussian(Scope([1]), mean=1.0, std=0.01), Gaussian(Scope([1]), mean=-5.0, std=0.01), Gaussian(Scope([1]), mean=0.0, std=0.01)],
            [Gaussian(Scope([2]), mean=10.0, std=0.01)]
        ]

        layer_spn = SPNSumNode(children=[
            SPNPartitionLayer(child_partitions=input_partitions)
            ],
            weights = [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]
        )
        
        nodes_spn = SPNSumNode(children=[SPNProductNode(children=[input_partitions[0][i], input_partitions[1][j], input_partitions[2][k]]) for (i,j,k) in itertools.product([0,1], [0,1,2], [0])],
            weights = [0.2, 0.1, 0.2, 0.2, 0.2, 0.1]
        )

        expected_mean = 0.2 * torch.tensor([3.0, 1.0, 10.0]) + 0.1 * torch.tensor([3.0, -5.0, 10.0]) + 0.2 * torch.tensor([3.0, 0.0, 10.0]) + 0.2 * torch.tensor([1.0, 1.0, 10.0]) + 0.2 * torch.tensor([1.0, -5.0, 10.0]) + 0.1 * torch.tensor([1.0, 0.0, 10.0])

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        self.assertTrue(torch.allclose(nodes_samples.mean(dim=0), expected_mean, atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(layer_samples.mean(dim=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))

        # sample from multiple outputs (with same scope)
        self.assertRaises(ValueError, sample, list(layer_spn.children())[0], 1, sampling_ctx=SamplingContext([0], [[0,1]]))

    def test_hadamard_layer_sampling(self):
        
        input_partitions = [
            [Gaussian(Scope([0]), mean=3.0, std=0.01), Gaussian(Scope([0]), mean=1.0, std=0.01)],
            [Gaussian(Scope([1]), mean=1.0, std=0.01), Gaussian(Scope([1]), mean=-5.0, std=0.01)],
            [Gaussian(Scope([2]), mean=10.0, std=0.01)]
        ]

        layer_spn = SPNSumNode(children=[
            SPNHadamardLayer(child_partitions=input_partitions)
            ],
            weights = [0.3, 0.7]
        )
        
        nodes_spn = SPNSumNode(children=[SPNProductNode(children=[input_partitions[0][i], input_partitions[1][j], input_partitions[2][k]]) for (i,j,k) in [[0,0,0], [1,1,0]]],
            weights = [0.3, 0.7]
        )

        expected_mean = 0.3 * torch.tensor([3.0, 1.0, 10.0]) + 0.7 * torch.tensor([1.0, -5.0, 10.0])

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        self.assertTrue(torch.allclose(nodes_samples.mean(axis=0), expected_mean, atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(layer_samples.mean(axis=0), nodes_samples.mean(axis=0), atol=0.01, rtol=0.1))
        
        # sample from multiple outputs (with same scope)
        self.assertRaises(ValueError, sample, list(layer_spn.children())[0], 1, sampling_ctx=SamplingContext([0], [[0,1]]))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()