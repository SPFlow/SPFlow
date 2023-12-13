import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import NegativeBinomial
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.negative_binomial import NegativeBinomial as BaseNegativeBinomial
from spflow.torch.structure.general.nodes.leaves.parametric.negative_binomial import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    n = random.randint(2, 10)
    p = random.random()

    torch_negative_binomial = NegativeBinomial(Scope([0]), n, p)
    node_negative_binomial = BaseNegativeBinomial(Scope([0]), n, p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(node_negative_binomial, data)
    log_probs_torch = log_likelihood(torch_negative_binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return
    n = random.randint(2, 10)
    p = random.random()

    torch_negative_binomial = NegativeBinomial(Scope([0]), n, p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs_torch = log_likelihood(torch_negative_binomial, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_negative_binomial.n.grad is None)
    tc.assertTrue(torch_negative_binomial.p_aux.grad is not None)

    n_orig = torch_negative_binomial.n.detach().clone()
    p_aux_orig = torch_negative_binomial.p_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(n_orig, torch_negative_binomial.n))
    tc.assertTrue(
        torch.allclose(
            p_aux_orig - torch_negative_binomial.p_aux.grad,
            torch_negative_binomial.p_aux,
        )
    )

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    if do_for_all_backends == "numpy":
        return


    torch.manual_seed(0)

    # initialize distribution
    torch_negative_binomial = NegativeBinomial(Scope([0]), 5, 0.3)

    # create dummy data
    p_target = 0.8
    data = torch.distributions.NegativeBinomial(5, 1 - p_target).sample((100000, 1))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(40):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_negative_binomial, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(
            torch_negative_binomial.p,
            tl.tensor(p_target, dtype=tl.float32),
            atol=1e-3,
            rtol=1e-3,
        )
    )

def test_likelihood_p_1(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # p = 1
    negative_binomial = NegativeBinomial(Scope([0]), 1, 1.0)

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(negative_binomial, data)
    log_probs = log_likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl_toNumpy(tl.exp(log_probs))))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets, atol=0.001, rtol=0.001))

def test_likelihood_n_float(do_for_all_backends):

    negative_binomial = NegativeBinomial(Scope([0]), 1, 0.5)
    tc.assertRaises(Exception, likelihood, negative_binomial, 0.5)

def test_likelihood_marginalization(do_for_all_backends):

    negative_binomial = NegativeBinomial(Scope([0]), 20, 0.3)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Negative Binomial distribution: integers N U {0}

    n = 20
    p = 0.3

    negative_binomial = NegativeBinomial(Scope([0]), n, p)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[-float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[float("inf")]]),
    )

    # check valid integers, but outside of valid range
    tc.assertRaises(ValueError, log_likelihood, negative_binomial, tl.tensor([[-1]]))

    # check valid integers within valid range
    log_likelihood(negative_binomial, tl.tensor([[0]]))
    log_likelihood(negative_binomial, tl.tensor([[100]]))

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[10.1]]),
    )

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)
    p = random.random()

    negative_binomial = NegativeBinomial(Scope([0]), n, p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(negative_binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            negative_binomial_updated = updateBackend(negative_binomial)
            log_probs_updated = log_likelihood(negative_binomial_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    n = random.randint(2, 10)
    p = random.random()

    node = NegativeBinomial(Scope([0]), n, p)
    dummy_data = tl.tensor(np.random.randint(1, n, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.randint(1, n, (3, 1)), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    n = random.randint(2, 10)
    p = random.random()

    node = NegativeBinomial(Scope([0]), n, p)
    dummy_data = tl.tensor(np.random.randint(1, n, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.randint(1, n, (3, 1)), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
