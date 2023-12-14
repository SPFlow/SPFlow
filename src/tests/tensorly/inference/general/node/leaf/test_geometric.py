import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import Geometric
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.node.leaf.geometric import Geometric as BaseGeometric
from spflow.torch.structure.general.node.leaf.geometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    p = random.random()

    torch_geometric = Geometric(Scope([0]), p)
    node_geometric = BaseGeometric(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, 10, (3, 1))

    log_probs = log_likelihood(node_geometric, data)
    log_probs_torch = log_likelihood(torch_geometric, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    p = random.random()

    torch_geometric = Geometric(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, 10, (3, 1))

    log_probs_torch = log_likelihood(torch_geometric, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_geometric.p_aux.grad is not None)

    p_aux_orig = torch_geometric.p_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(p_aux_orig - torch_geometric.p_aux.grad, torch_geometric.p_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_geometric.p, torch_geometric.dist.probs))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    torch_geometric = Geometric(Scope([0]), 0.3)

    # create dummy data
    p_target = 0.8
    data = torch.distributions.Geometric(p_target).sample((100000, 1)) + 1

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=0.9, momentum=0.6)

    # perform optimization (possibly overfitting)
    for i in range(40):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_geometric, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_geometric.p, tl.tensor(p_target, dtype=tl.float32), atol=1e-3, rtol=1e-3))

def test_likelihood_marginalization(do_for_all_backends):

    geometric = Geometric(Scope([0]), 0.5)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(geometric, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Geometric distribution: integers N\{0}

    geometric = Geometric(Scope([0]), 0.5)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        geometric,
        tl.tensor([[float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        geometric,
        tl.tensor([[-float("inf")]]),
    )

    # valid integers, but outside valid range
    tc.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[0.0]]))

    # valid integers within valid range
    data = tl.tensor([[1], [10]])

    probs = likelihood(geometric, data)
    log_probs = log_likelihood(geometric, data)

    tc.assertTrue(all(probs != 0.0))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl_toNumpy(tl.exp(log_probs))))

    # invalid floats
    tc.assertRaises(
        ValueError,
        log_likelihood,
        geometric,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(0.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        geometric,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(2.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, geometric, tl.tensor([[1.5]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    p = random.random()

    geometric = Geometric(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, 10, (3, 1))

    log_probs = log_likelihood(geometric, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            geometric_updated = updateBackend(geometric)
            log_probs_updated = log_likelihood(geometric_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    p = random.random()

    node = Geometric(Scope([0]), p)
    dummy_data = tl.tensor(np.random.randint(1, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.randint(1, 10, (3, 1)), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    p = random.random()

    node = Geometric(Scope([0]), p)
    dummy_data = tl.tensor(np.random.randint(1, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.randint(1, 10, (3, 1)), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
