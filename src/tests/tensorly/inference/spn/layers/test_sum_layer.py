import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.spn import SumLayer, SumNode
from spflow.tensorly.structure.spn.nodes.sum_node import toLayerBased
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.tensorly.inference.spn.nodes.sum_node import log_likelihood
from spflow.tensorly.structure.spn.layers_layerbased.sum_layer import toLayerBased
from spflow.tensorly.structure.spn.nodes.sum_node import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_sum_layer_likelihood(do_for_all_backends):

    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = SumNode(
        children=[
            SumLayer(
                n_nodes=3,
                children=input_nodes,
                weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
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

    layer_based_spn = toLayerBased(layer_spn)

    dummy_data = tl.tensor(
        [
            [1.0],
            [
                0.0,
            ],
            [0.25],
        ]
    )

    layer_ll = log_likelihood(layer_spn, dummy_data)
    nodes_ll = log_likelihood(nodes_spn, dummy_data)
    lb_ll = log_likelihood(layer_based_spn, dummy_data)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))
    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(tl.tensor(lb_ll, dtype=tl.float32))))

def test_sum_layer_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], dtype=tl.float32)

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

    # sum layer to be optimized
    sum_layer = SumLayer(n_nodes=3, children=[gaussian_1, gaussian_2], weights=weights)

    # make sure that weights are correctly projected
    tc.assertTrue(torch.allclose(weights, sum_layer.weights))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(sum_layer.parameters(), lr=0.5)

    for i in range(100):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log likelihood
        nll = -log_likelihood(sum_layer, data).mean()
        nll.backward()

        if i == 0:
            # check a few general things (just for the first update)

            # check if gradients are computed
            #tc.assertTrue(sum_layer._weights.grad is not None) # TODO: Sinnvolle abfrage?

            # update parameters
            optimizer.step()

            # verify that sum node weights are still valid after update
            tc.assertTrue(
                torch.allclose(
                    sum_layer.weights.sum(dim=-1),
                    tl.tensor([1.0, 1.0, 1.0],dtype=tl.float32),
                )
            )
        else:
            # update parameters
            optimizer.step()

    tc.assertTrue(
        torch.allclose(
            sum_layer.weights,
            tl.tensor([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]], dtype=tl.float32),
            atol=1e-3,
            rtol=1e-3,
        )
    )

def test_sum_layer_gradient_optimization_layerbased(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], dtype=tl.float32)

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

    # sum layer to be optimized
    sum_layer = SumLayer(n_nodes=3, children=[gaussian_1, gaussian_2], weights=weights)
    layer_based_spn = toLayerBased(sum_layer)

    # make sure that weights are correctly projected
    tc.assertTrue(torch.allclose(weights, layer_based_spn.weights))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(layer_based_spn.parameters(), lr=0.5)

    for i in range(100):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log likelihood
        nll = -log_likelihood(layer_based_spn, data).mean()
        nll.backward()

        if i == 0:
            # check a few general things (just for the first update)

            # check if gradients are computed
            #tc.assertTrue(sum_layer._weights.grad is not None) # TODO: Sinnvolle abfrage?

            # update parameters
            optimizer.step()

            # verify that sum node weights are still valid after update
            tc.assertTrue(
                torch.allclose(
                    layer_based_spn.weights.sum(dim=-1),
                    tl.tensor([1.0, 1.0, 1.0],dtype=tl.float32),
                )
            )
        else:
            # update parameters
            optimizer.step()

    tc.assertTrue(
        torch.allclose(
            layer_based_spn.weights,
            tl.tensor([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]], dtype=tl.float32),
            atol=1e-3,
            rtol=1e-3,
        )
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = SumNode(
        children=[
            SumLayer(
                n_nodes=3,
                children=input_nodes,
                weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
            ),
        ],
        weights=[0.3, 0.4, 0.3],
    )
    dummy_data = tl.tensor(
        [
            [1.0],
            [
                0.0,
            ],
            [0.25],
        ]
    )

    layer_ll = log_likelihood(layer_spn, dummy_data)
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer_spn)
            layer_ll_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(layer_ll_updated)))

def test_change_dtype(do_for_all_backends):
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = SumNode(
        children=[
            SumLayer(
                n_nodes=3,
                children=input_nodes,
                weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
            ),
        ],
        weights=[0.3, 0.4, 0.3],
    )
    dummy_data = tl.tensor(
        [
            [1.0],
            [
                0.0,
            ],
            [0.25],
        ]
    )

    layer_ll = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer_spn.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = SumNode(
        children=[
            SumLayer(
                n_nodes=3,
                children=input_nodes,
                weights=[[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.2, 0.7, 0.1]],
            ),
        ],
        weights=[0.3, 0.4, 0.3],
    )
    dummy_data = tl.tensor(
        [
            [1.0],
            [
                0.0,
            ],
            [0.25],
        ]
    )
    layer_ll = log_likelihood(layer_spn, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer_spn.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer_spn.to_device(cuda)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], device=cuda)
    layer_ll = log_likelihood(layer_spn, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")




if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
