import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import LogNormal
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.node.leaf.log_normal import LogNormal as BaseLogNormal
from spflow.torch.structure.general.node.leaf.log_normal import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero

    torch_log_normal = LogNormal(Scope([0]), mean, std)
    node_log_normal = BaseLogNormal(Scope([0]), mean, std)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(node_log_normal, data)
    log_probs_torch = log_likelihood(torch_log_normal, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero

    torch_log_normal = LogNormal(Scope([0]), mean, std)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs_torch = log_likelihood(torch_log_normal, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_log_normal.mean.grad is not None)
    tc.assertTrue(torch_log_normal.std_aux.grad is not None)

    mean_orig = torch_log_normal.mean.detach().clone()
    std_aux_orig = torch_log_normal.std_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(mean_orig - torch_log_normal.mean.grad, torch_log_normal.mean))
    tc.assertTrue(
        torch.allclose(
            std_aux_orig - torch_log_normal.std_aux.grad,
            torch_log_normal.std_aux,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_log_normal.mean, torch_log_normal.dist.loc))
    tc.assertTrue(torch.allclose(torch_log_normal.std, torch_log_normal.dist.scale))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_log_normal = LogNormal(Scope([0]), mean=1.0, std=2.0)

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.LogNormal(0.0, 1.0).sample((100000, 1))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_log_normal, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_log_normal.mean, tl.tensor(0.0, dtype=tl.float32), atol=1e-2, rtol=0.3))
    tc.assertTrue(torch.allclose(torch_log_normal.std, tl.tensor(1.0, dtype=tl.float32), atol=1e-2, rtol=0.3))

def test_likelihood_marginalization(do_for_all_backends):

    log_normal = LogNormal(Scope([0]), 0.0, 1.0)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(log_normal, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Log-Normal distribution: floats (0,inf)

    log_normal = LogNormal(Scope([0]), 0.0, 1.0)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        log_normal,
        tl.tensor([[float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        log_normal,
        tl.tensor([[-float("inf")]]),
    )

    # invalid float values
    tc.assertRaises(ValueError, log_likelihood, log_normal, tl.tensor([[0]]))

    # valid float values
    log_likelihood(
        log_normal,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    log_likelihood(log_normal, tl.tensor([[4.3]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero

    log_normal = LogNormal(Scope([0]), mean, std)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(log_normal, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            log_normal_updated = updateBackend(log_normal)
            log_probs_updated = log_likelihood(log_normal_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero

    node = LogNormal(Scope([0]), mean, std)
    dummy_data = tl.tensor(np.array([[5], [10]]), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.array([[5], [10]]), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    mean = random.random()
    std = random.random() + 1e-7  # offset by small number to avoid zero

    node = LogNormal(Scope([0]), mean, std)
    dummy_data = tl.tensor(np.array([[5], [10]]), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.array([[5], [10]]), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
