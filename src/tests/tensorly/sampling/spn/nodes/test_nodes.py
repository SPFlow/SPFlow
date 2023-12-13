import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.sampling import sample
from spflow.tensorly.structure.spn import Gaussian#, ProductNode, SumNode
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.structure.spn.nodes.sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_spn_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    s = SumNode(
        children=[
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -7.0, 1.0),
                            Gaussian(Scope([1]), 7.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -5.0, 1.0),
                            Gaussian(Scope([1]), 5.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.2, 0.8],
            ),
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -3.0, 1.0),
                            Gaussian(Scope([1]), 3.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -1.0, 1.0),
                            Gaussian(Scope([1]), 1.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.6, 0.4],
            ),
        ],
        weights=[0.7, 0.3],
    )

    samples = sample(s, 1000)
    expected_mean = 0.7 * (0.2 * tl.tensor([-7, 7]) + 0.8 * tl.tensor([-5, 5])) + 0.3 * (
        0.6 * tl.tensor([-3, 3]) + 0.4 * tl.tensor([-1, 1])
    )

    tc.assertTrue(np.allclose(tl.mean(samples, axis=0), expected_mean, rtol=0.1))

def test_sum_node_sampling(do_for_all_backends):

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    l1 = Gaussian(Scope([0]), -5.0, 1.0)
    l2 = Gaussian(Scope([0]), 5.0, 1.0)

    # ----- weights 0, 1 -----

    s = SumNode([l1, l2], weights=[0.001, 0.999])

    samples = sample(s, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), tl.tensor(5.0), rtol=0.1))

    # ----- weights 1, 0 -----

    s = SumNode([l1, l2], weights=[0.999, 0.001])

    samples = sample(s, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), tl.tensor(-5.0), rtol=0.1))

    # ----- weights 0.2, 0.8 -----

    s = SumNode([l1, l2], weights=[0.2, 0.8])

    samples = sample(s, 1000)
    tc.assertTrue(np.isclose(tl.mean(samples), tl.tensor(3.0), rtol=0.1))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    s = SumNode(
        children=[
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -7.0, 1.0),
                            Gaussian(Scope([1]), 7.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -5.0, 1.0),
                            Gaussian(Scope([1]), 5.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.2, 0.8],
            ),
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -3.0, 1.0),
                            Gaussian(Scope([1]), 3.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -1.0, 1.0),
                            Gaussian(Scope([1]), 1.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.6, 0.4],
            ),
        ],
        weights=[0.7, 0.3],
    )

    samples = sample(s, 1000)
    samples_mean = tl_toNumpy(samples).mean()
    for backend in backends:
        with tl.backend_context(backend):
            s_updated = updateBackend(s)
            samples_updated = sample(s_updated, 10000)
            samples_mean_updated = tl_toNumpy(samples_updated).mean()
            tc.assertTrue(np.allclose(samples_mean, samples_mean_updated, atol=0.1, rtol=0.1))


def test_change_dtype(do_for_all_backends):
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_default_dtype(torch.float32)

    layer = SumNode(
        children=[
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -7.0, 1.0),
                            Gaussian(Scope([1]), 7.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -5.0, 1.0),
                            Gaussian(Scope([1]), 5.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.2, 0.8],
            ),
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -3.0, 1.0),
                            Gaussian(Scope([1]), 3.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -1.0, 1.0),
                            Gaussian(Scope([1]), 1.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.6, 0.4],
            ),
        ],
        weights=[0.7, 0.3],
    )

    samples = sample(layer, 1000)
    tc.assertTrue(samples.dtype == tl.float32)
    layer.to_dtype(tl.float64)

    samples = sample(layer, 10000)
    tc.assertTrue(samples.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    layer = SumNode(
        children=[
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -7.0, 1.0),
                            Gaussian(Scope([1]), 7.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -5.0, 1.0),
                            Gaussian(Scope([1]), 5.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.2, 0.8],
            ),
            SumNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -3.0, 1.0),
                            Gaussian(Scope([1]), 3.0, 1.0),
                        ],
                    ),
                    ProductNode(
                        children=[
                            Gaussian(Scope([0]), -1.0, 1.0),
                            Gaussian(Scope([1]), 1.0, 1.0),
                        ],
                    ),
                ],
                weights=[0.6, 0.4],
            ),
        ],
        weights=[0.7, 0.3],
    )

    samples = sample(layer, 1000)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return

    tc.assertTrue(samples.device.type == "cpu")
    layer.to_device(cuda)

    samples = sample(layer, 10000)
    tc.assertTrue(samples.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
