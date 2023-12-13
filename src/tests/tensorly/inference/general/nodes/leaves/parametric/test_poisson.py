import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import Poisson
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.poisson import Poisson as BasePoisson
from spflow.torch.structure.general.nodes.leaves.parametric.poisson import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    l = random.randint(1, 10)

    torch_poisson = Poisson(Scope([0]), l)
    node_poisson = BasePoisson(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs = log_likelihood(node_poisson, data)
    log_probs_torch = log_likelihood(torch_poisson, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    l = random.randint(1, 10)

    torch_poisson = Poisson(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs_torch = log_likelihood(torch_poisson, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_poisson.l_aux.grad is not None)

    l_aux_orig = torch_poisson.l_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_poisson.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(l_aux_orig - torch_poisson.l_aux.grad, torch_poisson.l_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_poisson.l, torch_poisson.dist.rate))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_poisson = Poisson(Scope([0]), l=1.0)

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Poisson(rate=4.0).sample((100000, 1))

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

    tc.assertTrue(torch.allclose(torch_poisson.l, tl.tensor(4.0, dtype=tl.float32), atol=1e-3, rtol=0.3))

def test_likelihood_marginalization(do_for_all_backends):

    poisson = Poisson(Scope([0]), 1.0)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(poisson, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Poisson distribution: integers N U {0}

    l = random.random()

    poisson = Poisson(Scope([0]), l)

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-float("inf")]]))
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[float("inf")]]))

    # check valid integers, but outside of valid range
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-1]]))

    # check valid integers within valid range
    log_likelihood(poisson, tl.tensor([[0]]))
    log_likelihood(poisson, tl.tensor([[100]]))

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        poisson,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        poisson,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[10.1]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.randint(1, 10)

    poisson = Poisson(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs = log_likelihood(poisson, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            poisson_updated = updateBackend(poisson)
            log_probs_updated = log_likelihood(poisson_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    l = random.randint(1, 10)

    node = Poisson(Scope([0]), l)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    l = random.randint(1, 10)

    node = Poisson(Scope([0]), l)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
