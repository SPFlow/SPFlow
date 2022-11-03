from spflow.meta.dispatch.sampling_context import SamplingContext
from spflow.meta.data.scope import Scope
from spflow.torch.structure.nodes.leaves.parametric.cond_binomial import (
    CondBinomial,
)
from spflow.torch.inference.nodes.leaves.parametric.cond_binomial import (
    log_likelihood,
)
from spflow.torch.sampling.nodes.leaves.parametric.cond_binomial import sample
from spflow.torch.sampling.spn.nodes.node import sample
from spflow.torch.structure.layers.leaves.parametric.cond_binomial import (
    CondBinomialLayer,
)
from spflow.torch.inference.layers.leaves.parametric.cond_binomial import (
    log_likelihood,
)
from spflow.torch.sampling.layers.leaves.parametric.cond_binomial import sample
from spflow.torch.sampling.spn.layers.sum_layer import sample
from spflow.torch.inference.module import log_likelihood
from spflow.torch.sampling.module import sample

import torch
import numpy as np
import random
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

        # set seed
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        layer = CondBinomialLayer(
            scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
            n=[3, 2, 3],
            cond_f=lambda data: {"p": [0.2, 0.8, 0.2]},
        )

        nodes = [
            CondBinomial(Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.2}),
            CondBinomial(Scope([1], [2]), n=2, cond_f=lambda data: {"p": 0.8}),
            CondBinomial(Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.2}),
        ]

        # make sure sampling fron non-overlapping scopes works
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[0, 1]]))
        sample(layer, 1, sampling_ctx=SamplingContext([0], [[2, 1]]))
        # make sure sampling from overlapping scopes does not works
        self.assertRaises(
            ValueError,
            sample,
            layer,
            1,
            sampling_ctx=SamplingContext([0], [[0, 2]]),
        )
        self.assertRaises(
            ValueError,
            sample,
            layer,
            1,
            sampling_ctx=SamplingContext([0], [[]]),
        )

        layer_samples = sample(
            layer,
            10000,
            sampling_ctx=SamplingContext(
                list(range(10000)),
                [[0, 1] for _ in range(5000)]
                + [[2, 1] for _ in range(5000, 10000)],
            ),
        )
        nodes_samples = torch.concat(
            [
                torch.cat(
                    [sample(nodes[0], 5000), sample(nodes[2], 5000)], dim=0
                ),
                sample(nodes[1], 10000)[:, [1]],
            ],
            dim=1,
        )

        expected_mean = torch.tensor([3 * 0.2, 2 * 0.8])
        self.assertTrue(
            torch.allclose(
                nodes_samples.mean(dim=0), expected_mean, atol=0.01, rtol=0.1
            )
        )
        self.assertTrue(
            torch.allclose(
                layer_samples.mean(dim=0),
                nodes_samples.mean(dim=0),
                atol=0.01,
                rtol=0.1,
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
