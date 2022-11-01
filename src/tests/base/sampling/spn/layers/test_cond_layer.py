from spflow.meta.dispatch.sampling_context import SamplingContext
from spflow.meta.data.scope import Scope
from spflow.base.structure.spn.nodes.node import SPNSumNode
from spflow.base.inference.spn.nodes.node import log_likelihood
from spflow.base.sampling.spn.nodes.node import sample
from spflow.base.structure.spn.layers.cond_layer import SPNCondSumLayer
from spflow.base.inference.spn.layers.cond_layer import log_likelihood
from spflow.base.sampling.spn.layers.cond_layer import sample
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.gaussian import sample
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample

import numpy as np
import random
import unittest
import itertools


class TestNode(unittest.TestCase):
    def test_sum_layer_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        input_nodes = [
            Gaussian(Scope([0]), mean=3.0, std=0.01),
            Gaussian(Scope([0]), mean=1.0, std=0.01),
            Gaussian(Scope([0]), mean=0.0, std=0.01),
        ]

        layer_spn = SPNSumNode(
            children=[
                SPNCondSumLayer(
                    n_nodes=3,
                    children=input_nodes,
                    cond_f=lambda data: {
                        "weights": [
                            [0.8, 0.1, 0.1],
                            [0.2, 0.3, 0.5],
                            [0.2, 0.7, 0.1],
                        ]
                    },
                ),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        nodes_spn = SPNSumNode(
            children=[
                SPNSumNode(children=input_nodes, weights=[0.8, 0.1, 0.1]),
                SPNSumNode(children=input_nodes, weights=[0.2, 0.3, 0.5]),
                SPNSumNode(children=input_nodes, weights=[0.2, 0.7, 0.1]),
            ],
            weights=[0.3, 0.4, 0.3],
        )

        layer_samples = sample(layer_spn, 10000)
        nodes_samples = sample(nodes_spn, 10000)

        expected_mean = (
            0.3 * (0.8 * 3.0 + 0.1 * 1.0 + 0.1 * 0.0)
            + 0.4 * (0.2 * 3.0 + 0.3 * 1.0 + 0.5 * 0.0)
            + 0.3 * (0.2 * 3.0 + 0.7 * 1.0 + 0.1 * 0.0)
        )
        self.assertTrue(
            np.allclose(
                nodes_samples.mean(axis=0),
                np.array([expected_mean]),
                atol=0.01,
                rtol=0.1,
            )
        )
        self.assertTrue(
            np.allclose(
                layer_samples.mean(axis=0),
                nodes_samples.mean(axis=0),
                atol=0.01,
                rtol=0.1,
            )
        )

        # sample from multiple outputs (with same scope)
        self.assertRaises(
            ValueError,
            sample,
            layer_spn.children[0],
            1,
            sampling_ctx=SamplingContext([0], [[0, 1]]),
        )


if __name__ == "__main__":
    unittest.main()
