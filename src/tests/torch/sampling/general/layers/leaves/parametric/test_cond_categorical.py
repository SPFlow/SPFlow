import random
import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext
from spflow.torch.inference import log_likelihood
from spflow.torch.sampling import sample
from spflow.torch.structure.spn import CondCategorical, CondCategoricalLayer


class TestCategorical(unittest.TestCase):

    def test_sampling(self):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        scopes = [Scope([0], [2]), Scope([1], [2]), Scope([0], [2])]
        ks = [2, 2, 2]
        ps = [[0.5, 0.5], [0.3, 0.7], [0.5, 0.5]]

        layer = CondCategoricalLayer(
            scope=scopes,
            cond_f=lambda data: {"k": ks, "p": ps},
        )

        nodes = [
            CondCategorical(scope=scopes[0], cond_f=lambda data: {"k": ks[0], "p": ps[0]}),
            CondCategorical(scope=scopes[1], cond_f=lambda data: {"k": ks[1], "p": ps[1]}),
            CondCategorical(scope=scopes[2], cond_f=lambda data: {"k": ks[2], "p": ps[2]})
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
                [[0, 1] for _ in range(5000)] + [[2, 1] for _ in range(5000, 10000)],
            ),
        )
        nodes_samples = torch.concat(
            [
                torch.cat([sample(nodes[0], 5000), sample(nodes[2], 5000)], dim=0),
                sample(nodes[1], 10000)[:, [1]],
            ],
            dim=1,
        )

        expected_mean = torch.tensor([[0.5, 0.7]])
        self.assertTrue(torch.allclose(nodes_samples.mean(dim=0), expected_mean, atol=0.01, rtol=0.1))
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
