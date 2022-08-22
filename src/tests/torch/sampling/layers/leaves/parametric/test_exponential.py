from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.meta.scope.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.exponential import Exponential
from spflow.torch.inference.nodes.leaves.parametric.exponential import log_likelihood
from spflow.torch.sampling.nodes.leaves.parametric.exponential import sample
from spflow.torch.sampling.nodes.node import sample
from spflow.torch.structure.layers.leaves.parametric.exponential import ExponentialLayer
from spflow.torch.inference.layers.leaves.parametric.exponential import log_likelihood
from spflow.torch.sampling.layers.leaves.parametric.exponential import sample
from spflow.torch.sampling.layers.layer import sample
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

    def test_layer_sampling(self):

        layer = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.8, 0.2])

        nodes = [
                Exponential(Scope([0]), l=0.2),
                Exponential(Scope([1]), l=1.8),
                Exponential(Scope([0]), l=0.2)
        ]

        # make sure sampling fron non-overlapping scopes works
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[0,1]]))
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[2,1]]))
        # make sure sampling from overlapping scopes does not works
        self.assertRaises(ValueError, sample, layer, 1, sampling_ctx=SamplingContext([0], [[0,2]]))
        self.assertRaises(ValueError, sample, layer, 1, sampling_ctx=SamplingContext([0], [[]]))

        layer_samples = sample(layer, 10000, sampling_ctx=SamplingContext(list(range(10000)), [[0,1] for _ in range(5000)] + [[2,1] for _ in range(5000, 10000)]))
        nodes_samples = torch.concat([
                torch.cat([sample(nodes[0], 5000), sample(nodes[2], 5000)], dim=0),
                sample(nodes[1], 10000)[:, [1]]
            ], dim=1)

        expected_mean = torch.tensor([1.0/0.2, 1.0/1.8])
        self.assertTrue(torch.allclose(nodes_samples.mean(dim=0), expected_mean, atol=0.01, rtol=0.1))
        self.assertTrue(torch.allclose(layer_samples.mean(dim=0), nodes_samples.mean(dim=0), atol=0.01, rtol=0.1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()