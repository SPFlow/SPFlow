import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_poisson import PoissonLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_poisson import Poisson
from spflow.torch.structure.general.layers.leaves.parametric.poisson import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

    nodes = [
        Poisson(Scope([0]), l=0.2),
        Poisson(Scope([1]), l=1.0),
        Poisson(Scope([0]), l=2.3),
    ]

    dummy_data = tl.tensor([[1, 3], [3, 7], [2, 1]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    l = tl.tensor([random.randint(1, 10), random.randint(1, 10)])

    torch_poisson = PoissonLayer(scope=[Scope([0]), Scope([1])], l=l)

    # create dummy input data (batch size x random variables)
    data = torch.cat([torch.randint(0, 10, (3, 1)), torch.randint(0, 10, (3, 1))], dim=1)

    log_probs_torch = log_likelihood(torch_poisson, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_poisson.l_aux.grad is not None)

    l_aux_orig = torch_poisson.l_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_poisson.l, torch_poisson.dist().rate))

def test_gradient_optimization(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_poisson = PoissonLayer(scope=[Scope([0]), Scope([1])], l=1.0)

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Poisson(rate=4.0).sample((100000, 2))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=0.1)

    # perform optimization (possibly overfitting)
    for i in range(40):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_poisson, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_poisson.l, tl.tensor([4.0, 4.0], dtype=tl.float32), atol=1e-3, rtol=0.3))

def test_likelihood_marginalization(do_for_all_backends):

    poisson = PoissonLayer(scope=[Scope([0]), Scope([1])], l=random.random() + 1e-7)
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(poisson, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

    dummy_data = tl.tensor([[1, 3], [3, 7], [2, 1]])

    layer_ll = log_likelihood(layer, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    layer = PoissonLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]])
    layer_ll = log_likelihood(layer, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer.to_device(cuda)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], device=cuda)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
