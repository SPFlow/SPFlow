import random
import unittest

import numpy as np
import tensorly as tl
from spflow.tensorly.utils.helper_functions import tl_allclose

from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import CondSumLayer, SumNode
from spflow.tensorly.structure.general.nodes.leaves import Gaussian
from spflow.meta.data import Scope
from spflow.meta.dispatch import SamplingContext


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

        layer_spn = SumNode(
            children=[
                CondSumLayer(
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

        nodes_spn = SumNode(
            children=[
                SumNode(children=input_nodes, weights=[0.8, 0.1, 0.1]),
                SumNode(children=input_nodes, weights=[0.2, 0.3, 0.5]),
                SumNode(children=input_nodes, weights=[0.2, 0.7, 0.1]),
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
            tl_allclose(
                nodes_samples.mean(axis=0),
                tl.tensor([expected_mean]),
                atol=0.01,
                rtol=0.1,
            )
        )
        self.assertTrue(
            tl_allclose(
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
