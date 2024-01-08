import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope

from spflow.modules.module import log_likelihood
from spflow.modules.node import Gaussian
from spflow.modules.layer import CondSumLayer
from spflow.modules.node import CondSumNode, log_likelihood
from spflow.modules.node import toLayerBased
from spflow.modules.node import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_sum_layer_likelihood(do_for_all_backends):
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = CondSumNode(
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
        cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
    )

    nodes_spn = CondSumNode(
        children=[
            CondSumNode(
                children=input_nodes,
                cond_f=lambda data: {"weights": [0.8, 0.1, 0.1]},
            ),
            CondSumNode(
                children=input_nodes,
                cond_f=lambda data: {"weights": [0.2, 0.3, 0.5]},
            ),
            CondSumNode(
                children=input_nodes,
                cond_f=lambda data: {"weights": [0.2, 0.7, 0.1]},
            ),
        ],
        cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
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

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))
    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(tl.tensor(lb_ll, dtype=tl.float32))))


def test_sum_layer_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], requires_grad=True)

    data_1 = torch.randn((70000, 1))
    data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
    data_2 = torch.randn((30000, 1))
    data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

    data = torch.cat([data_1, data_2])

    # initialize Gaussians
    gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
    gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

    # sum layer to be optimized
    sum_layer = CondSumLayer(
        n_nodes=3,
        children=[gaussian_1, gaussian_2],
        cond_f=lambda data: {"weights": weights},
    )

    ll = log_likelihood(sum_layer, data).mean()
    ll.backward()

    tc.assertTrue(weights.grad is not None)


def test_sum_layer_gradient_computation2(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return
    torch.manual_seed(0)

    # generate random weights for a sum node with two children
    weights = tl.tensor([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], requires_grad=True)

    data_1 = torch.randn((70000, 1))
    data_1 = (data_1 - data_1.mean()) / data_1.std() + 5.0
    data_2 = torch.randn((30000, 1))
    data_2 = (data_2 - data_2.mean()) / data_2.std() - 5.0

    data = torch.cat([data_1, data_2])

    # initialize Gaussians
    gaussian_1 = Gaussian(Scope([0]), 5.0, 1.0)
    gaussian_2 = Gaussian(Scope([0]), -5.0, 1.0)

    # sum layer to be optimized
    sum_layer = CondSumLayer(
        n_nodes=3,
        children=[gaussian_1, gaussian_2],
        cond_f=lambda data: {"weights": weights},
    )
    # transform node_based model to layer_based
    layer_based_spn = toLayerBased(sum_layer)

    ll = log_likelihood(layer_based_spn, data).mean()
    ll.backward()

    tc.assertTrue(weights.grad is not None)


def test_update_backend(do_for_all_backends):
    backends = ["numpy"]
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = CondSumNode(
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
        cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
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
            tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(layer_ll_updated)))


def test_change_dtype(do_for_all_backends):
    input_nodes = [
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
        Gaussian(Scope([0])),
    ]

    layer_spn = CondSumNode(
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
        cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
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

    layer_spn = CondSumNode(
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
        cond_f=lambda data: {"weights": [0.3, 0.4, 0.3]},
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
