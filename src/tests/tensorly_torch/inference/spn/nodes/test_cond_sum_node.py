import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import CondSumNode, ProductNode
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn.nodes.cond_sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

def create_example_spn():
    spn = CondSumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    CondSumNode(
                        children=[
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                            ProductNode(
                                children=[
                                    Gaussian(Scope([1])),
                                    Gaussian(Scope([2])),
                                ]
                            ),
                        ],
                        cond_f=lambda data: {"weights": tl.tensor([0.3, 0.7])},
                    ),
                ],
            ),
            ProductNode(
                children=[
                    ProductNode(
                        children=[
                            Gaussian(Scope([0])),
                            Gaussian(Scope([1])),
                        ]
                    ),
                    Gaussian(Scope([2])),
                ]
            ),
        ],
        cond_f=lambda data: {"weights": tl.tensor([0.4, 0.6])},
    )
    return spn


tc = unittest.TestCase()

def test_likelihood(do_for_all_backends):
    dummy_spn = create_example_spn()
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]])

    l_result = likelihood(dummy_spn, dummy_data)
    ll_result = log_likelihood(dummy_spn, dummy_data)
    tc.assertTrue(np.isclose(tl_toNumpy(l_result[0][0]), tl.tensor(0.023358)))
    tc.assertTrue(np.isclose(tl_toNumpy(ll_result[0][0]), tl.tensor(-3.7568156)))

"""
def test_likelihood_marginalization(do_for_all_backends):
    spn = create_example_spn()
    dummy_data = tl.tensor([[float("nan"), 0.0, 1.0]])

    l_result = likelihood(spn, dummy_data)
    ll_result = log_likelihood(spn, dummy_data)
    tc.assertAlmostEqual(l_result[0][0], 0.09653235)
    tc.assertAlmostEqual(ll_result[0][0], -2.33787707)
"""

def test_likelihood_marginalization(do_for_all_backends):
    spn = create_example_spn()
    dummy_data = tl.tensor([[float("nan"), 0.0, 1.0]])

    l_result = likelihood(spn, dummy_data)
    ll_result = log_likelihood(spn, dummy_data)
    tc.assertTrue(np.isclose(tl_toNumpy(l_result[0][0]), tl.tensor(0.09653235)))
    tc.assertTrue(np.isclose(tl_toNumpy(ll_result[0][0]), tl.tensor(-2.33787707)))

def test_sum_node_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([0.3, 0.7], requires_grad=True)

    data_1 = torch.randn((70000, 1))
    data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
    data_2 = torch.randn((30000, 1))
    data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

    data = torch.cat([data_1, data_2])

    # initialize Gaussians
    gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
    gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

    # sum node to be optimized
    sum_node = CondSumNode(
        children=[gaussian_1, gaussian_2],
        cond_f=lambda data: {"weights": weights},
    )

    ll = log_likelihood(sum_node, data).mean()
    ll.backward()

    tc.assertTrue(weights.grad is not None)

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    dummy_spn = create_example_spn()
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]])

    ll_result = log_likelihood(dummy_spn, dummy_data)

    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(dummy_spn)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            tc.assertTrue(np.allclose(tl_toNumpy(ll_result), tl_toNumpy(layer_ll_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
