import random
import unittest

import numpy as np
import torch
from packaging import version
import tensorly as tl

from spflow.tensorly.structure.spn import Exponential
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.exponential import Exponential as BaseExponential
from spflow.torch.structure.general.nodes.leaves.parametric.exponential import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    l = random.random() + 1e-7  # small offset to avoid zero

    torch_exponential = Exponential(Scope([0]), l)
    node_exponential = BaseExponential(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(node_exponential, data)
    log_probs_torch = log_likelihood(torch_exponential, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    l = random.random()

    torch_exponential = Exponential(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs_torch = log_likelihood(torch_exponential, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_exponential.l_aux.grad is not None)

    l_aux_orig = torch_exponential.l_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        torch.allclose(
            l_aux_orig - torch_exponential.l_aux.grad,
            torch_exponential.l_aux,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist.rate))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_exponential = Exponential(Scope([0]), l=0.5)

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Exponential(rate=1.5).sample((100000, 1))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_exponential, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_exponential.l, tl.tensor(1.5, dtype=tl.float32), atol=1e-3, rtol=0.3))

def test_likelihood_marginalization(do_for_all_backends):

    exponential = Exponential(Scope([0]), 1.0)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(exponential, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Exponential distribution: floats [0,inf) (note: 0 excluded in pytorch support)

    l = 1.5
    exponential = Exponential(Scope([0]), l)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[-float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[float("inf")]]),
    )

    # check valid float values (within range)
    log_likelihood(
        exponential,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    log_likelihood(exponential, tl.tensor([[10.5]]))

    # check invalid float values (outside range)
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )

    if version.parse(torch.__version__) < version.parse("1.11.0"):
        # edge case 0 (part of the support in scipy, but NOT pytorch)
        tc.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[0.0]]))
    else:
        # edge case 0
        log_likelihood(exponential, tl.tensor([[0.0]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.random() + 1e-7  # small offset to avoid zero

    exponential = Exponential(Scope([0]), l)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(exponential, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            exponential_updated = updateBackend(exponential)
            log_probs_updated = log_likelihood(exponential_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    l = random.random() + 1e-7  # small offset to avoid zero

    node = Exponential(Scope([0]), l)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    l = random.random() + 1e-7  # small offset to avoid zero

    node = Exponential(Scope([0]), l)
    dummy_data = tl.tensor(np.random.rand(3, 1), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.rand(3, 1), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
