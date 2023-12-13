import unittest
from itertools import chain
import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.structure.spn import ProductNode, SumNode
from spflow.tensorly.utils.projections import proj_convex_to_real, proj_real_to_convex
from spflow.tensorly.structure.spn.nodes.sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

from ....structure.general.nodes.dummy_node import DummyNode


def create_example_spn():
    spn = SumNode(
        children=[
            ProductNode(
                children=[
                    Gaussian(Scope([0])),
                    SumNode(
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
                        weights=tl.tensor([0.3, 0.7]),
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
        weights=tl.tensor([0.4, 0.6]),
    )
    return spn


tc = unittest.TestCase()

def test_likelihood(do_for_all_backends):
    dummy_spn = create_example_spn()
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]])

    l_result = likelihood(dummy_spn, dummy_data)
    ll_result = log_likelihood(dummy_spn, dummy_data)
    tc.assertTrue(np.isclose(tl_toNumpy(tl.tensor(l_result[0][0], dtype=tl.float32)), tl.tensor(0.023358, dtype=tl.float32)))
    tc.assertTrue(np.isclose(tl_toNumpy(tl.tensor(ll_result[0][0], dtype=tl.float32)), tl.tensor(-3.7568156, dtype=tl.float32)))

def test_likelihood_marginalization(do_for_all_backends):
    spn = create_example_spn()
    dummy_data = tl.tensor([[float("nan"), 0.0, 1.0]])

    l_result = likelihood(spn, dummy_data)
    ll_result = log_likelihood(spn, dummy_data)
    tc.assertTrue(np.isclose(tl_toNumpy(tl.tensor(l_result[0][0], dtype=tl.float32)), tl.tensor(0.09653235, dtype=tl.float32)))
    tc.assertTrue(np.isclose(tl_toNumpy(tl.tensor(ll_result[0][0], dtype=tl.float32)), tl.tensor(-2.33787707, dtype=tl.float32)))

def test_dummy_node_likelihood_not_implemented(do_for_all_backends):
    dummy_node = DummyNode()
    dummy_data = tl.tensor([[1.0]])

    tc.assertRaises(NotImplementedError, log_likelihood, dummy_node, dummy_data)
    tc.assertRaises(NotImplementedError, likelihood, dummy_node, dummy_data)

def test_sum_node_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([0.3, 0.7])

    data_1 = torch.randn((70000, 1))
    data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
    data_2 = torch.randn((30000, 1))
    data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

    data = torch.cat([data_1, data_2])

    # initialize Gaussians
    gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
    gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

    # freeze Gaussians
    gaussian_1.requires_grad = False
    gaussian_2.requires_grad = False

    # sum node to be optimized
    sum_node = SumNode(
        children=[gaussian_1, gaussian_2],
        weights=weights,
    )
    # make sure that weights are correctly projected
    tc.assertTrue(torch.allclose(weights, sum_node.weights))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(sum_node.parameters(), lr=0.5)

    for i in range(50):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log likelihood
        nll = -log_likelihood(sum_node, data).mean()
        nll.backward()

        if i == 0:
            # check a few general things (just for the first update)

            # check if gradients are computed
            tc.assertTrue(sum_node._weights.grad is not None)

            # update parameters
            optimizer.step()

            # verify that sum node weights are still valid after update
            tc.assertTrue(torch.isclose(sum_node.weights.sum(), tl.tensor(1.0)))
        else:
            # update parameters
            optimizer.step()

    tc.assertTrue(torch.allclose(sum_node.weights, tl.tensor([0.7, 0.3]), atol=1e-3, rtol=1e-3))

def test_projection(do_for_all_backends):
    torch.set_default_dtype(torch.float64)
    tc.assertTrue(np.allclose(tl.sum(proj_real_to_convex(tl.randn((5,)))), tl.tensor(1.0)))

    weights = tl.random.random_tensor((5,))
    weights /= tl.sum(weights)

    tc.assertTrue(np.allclose(proj_convex_to_real(weights), tl.log(weights)))

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

def test_change_dtype(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    layer_spn = create_example_spn()
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]], dtype=tl.float32)

    layer_ll = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer_spn.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):

    cuda = torch.device("cuda")
    layer_spn = create_example_spn()
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]])

    layer_ll = log_likelihood(layer_spn, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer_spn.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer_spn.to_device(cuda)
    dummy_data = tl.tensor([[1.0, 0.0, 1.0]], device=cuda)
    layer_ll = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
