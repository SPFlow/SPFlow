import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.spn import Bernoulli
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.bernoulli import Bernoulli as BaseBernoulli
from spflow.torch.structure.general.nodes.leaves.parametric.bernoulli import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    p = random.random()

    general_bernoulli = Bernoulli(Scope([0]), p)
    node_bernoulli = BaseBernoulli(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))

    log_probs = log_likelihood(node_bernoulli, data)
    log_probs_general = log_likelihood(general_bernoulli, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_general)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    p = random.random()

    torch_bernoulli = Bernoulli(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))

    log_probs_torch = log_likelihood(torch_bernoulli, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_bernoulli.p_aux.grad is not None)

    p_aux_orig = torch_bernoulli.p_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(p_aux_orig - torch_bernoulli.p_aux.grad, torch_bernoulli.p_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_bernoulli.p, torch_bernoulli.dist.probs))

def test_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    torch_bernoulli = Bernoulli(Scope([0]), 0.3)

    # create dummy data
    p_target = 0.8
    data = torch.bernoulli(torch.full((100000, 1), p_target))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_bernoulli.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(40):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_bernoulli, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_bernoulli.p, tl.tensor(p_target, dtype=tl.float64), atol=1e-3, rtol=1e-3))

def test_likelihood_p_0(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    # p = 0
    bernoulli = Bernoulli(Scope([0]), 0.0)

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl_toNumpy(tl.exp(log_probs))))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_p_1(do_for_all_backends):

    # p = 1
    bernoulli = Bernoulli(Scope([0]), 1.0)

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[0.0], [1.0]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl_toNumpy(tl.exp(log_probs))))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_marginalization(do_for_all_backends):

    bernoulli = Bernoulli(Scope([0]), random.random())
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Bernoulli distribution: integers {0,1}

    p = random.random()
    bernoulli = Bernoulli(Scope([0]), p)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[-float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[float("inf")]]),
    )

    # check valid integers inside valid range
    log_likelihood(bernoulli, tl.tensor([[0.0], [1.0]]))

    # check valid integers, but outside of valid range
    tc.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[-1]]))
    tc.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[2]]))

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(2.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        bernoulli,
        tl.tensor([[np.nextafter(tl.tensor(1.0), tl.tensor(0.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[0.5]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    p = random.random()

    bernoulli = Bernoulli(Scope([0]), p)

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))

    log_probs = log_likelihood(bernoulli, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            bernoulli_updated = updateBackend(bernoulli)
            log_probs_updated = log_likelihood(bernoulli_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
