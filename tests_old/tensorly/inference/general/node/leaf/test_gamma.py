import random
import unittest

import numpy as np
import torch
from packaging import version
import tensorly as tl

from spflow.structure.spn import Gamma
from spflow.meta.data import Scope
from spflow.modules.module import log_likelihood, likelihood
from spflow.base.structure.general.node.leaf.gamma import Gamma as BaseGamma
from spflow.torch.structure.general.node.leaf.gaussian import updateBackend
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_inference(do_for_all_backends):
    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)

    torch_gamma = Gamma(Scope([0]), alpha, beta)
    node_gamma = BaseGamma(Scope([0]), alpha, beta)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(node_gamma, data)
    log_probs_torch = log_likelihood(torch_gamma, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tle.toNumpy(log_probs_torch)))


def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)

    torch_gamma = Gamma(Scope([0]), alpha, beta)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs_torch = log_likelihood(torch_gamma, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_gamma.alpha_aux.grad is not None)
    tc.assertTrue(torch_gamma.beta_aux.grad is not None)

    alpha_aux_orig = torch_gamma.alpha_aux.detach().clone()
    beta_aux_orig = torch_gamma.beta_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        torch.allclose(
            alpha_aux_orig - torch_gamma.alpha_aux.grad,
            torch_gamma.alpha_aux,
        )
    )
    tc.assertTrue(torch.allclose(beta_aux_orig - torch_gamma.beta_aux.grad, torch_gamma.beta_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_gamma.alpha, torch_gamma.dist.concentration))
    tc.assertTrue(torch.allclose(torch_gamma.beta, torch_gamma.dist.rate))


def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_gamma = Gamma(Scope([0]), alpha=1.0, beta=2.0)

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Gamma(concentration=2.0, rate=1.0).sample((100000, 1))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):
        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_gamma, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_gamma.alpha, tl.tensor(2.0, dtype=tl.float32), atol=1e-3, rtol=0.3))
    tc.assertTrue(torch.allclose(torch_gamma.beta, tl.tensor(1.0, dtype=tl.float32), atol=1e-3, rtol=0.3))


def test_marginalization(do_for_all_backends):
    gamma = Gamma(Scope([0]), 1.0, 1.0)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(gamma, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor(1.0)))


def test_support(do_for_all_backends):
    # Support for Gamma distribution: floats (0,inf)

    # TODO:
    #   likelihood:     x=0 -> POS_EPS (?)
    #   log-likelihood: x=0 -> POS_EPS (?)

    gamma = Gamma(Scope([0]), 1.0, 1.0)

    # TODO: 0

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[-float("inf")]]))
    tc.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[float("inf")]]))

    # check finite values > 0
    log_likelihood(
        gamma,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    log_likelihood(gamma, tl.tensor([[10.5]]))

    data = tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]])

    probs = likelihood(gamma, data)
    log_probs = log_likelihood(gamma, data)

    tc.assertTrue(all(data != 0.0))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), tle.toNumpy(tl.exp(log_probs))))

    # check invalid float values (outside range)
    if version.parse(torch.__version__) < version.parse("1.12.0") or do_for_all_backends == "numpy":
        # edge case 0
        tc.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[0.0]]))
    else:
        # edge case 0
        log_likelihood(gamma, tl.tensor([[0.0]]))

    tc.assertRaises(
        ValueError,
        log_likelihood,
        gamma,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)

    gamma = Gamma(Scope([0]), alpha, beta)

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(gamma, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            gamma_updated = updateBackend(gamma)
            log_probs_updated = log_likelihood(gamma_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(log_probs), tle.toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)

    node = Gamma(Scope([0]), alpha, beta)
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
    alpha = random.randint(1, 5)
    beta = random.randint(1, 5)

    node = Gamma(Scope([0]), alpha, beta)
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