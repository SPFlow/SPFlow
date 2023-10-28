import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import CondGeometric
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.torch.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.cond_geometric import CondGeometric as BaseCondGeometric
from spflow.torch.structure.general.nodes.leaves.parametric.cond_geometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):

    p = random.random()
    cond_f = lambda data: {"p": 0.5}

    geometric = CondGeometric(Scope([0], [1]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[1], [5], [10]])
    targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

    probs = likelihood(geometric, data)
    log_probs = log_likelihood(geometric, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    geometric = CondGeometric(Scope([0], [1]))

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[geometric] = {"p": 0.5}

    # create test inputs/outputs
    data = tl.tensor([[1], [5], [10]])
    targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

    probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    geometric = CondGeometric(Scope([0], [1]))

    cond_f = lambda data: {"p": 0.5}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[geometric] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[1], [5], [10]])
    targets = tl.tensor([[0.5], [0.03125], [0.000976563]])

    probs = likelihood(geometric, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(geometric, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):

    p = random.random()

    torch_geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": p})
    node_geometric = BaseCondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, 10, (3, 1))

    log_probs = log_likelihood(node_geometric, data)
    log_probs_torch = log_likelihood(torch_geometric, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    p = tl.tensor(random.random(), requires_grad=True)

    torch_geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, 10, (3, 1))

    log_probs_torch = log_likelihood(torch_geometric, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(p.grad is not None)

def test_likelihood_marginalization(do_for_all_backends):

    geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(geometric, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Geometric distribution: integers N\{0}

    geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": 0.5})

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
    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))

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

    geometric = CondGeometric(Scope([0], [1]), cond_f=lambda data: {"p": p})

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


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
