import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import CondBinomial
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.cond_binomial import CondBinomial as BaseCondBinomial
from spflow.torch.structure.general.nodes.leaves.parametric.cond_binomial import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy, tl_unsqueeze

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):

    p = random.random()
    cond_f = lambda data: {"p": p}

    binomial = CondBinomial(Scope([0], [1]), n=1, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(binomial, data)
    log_probs = log_likelihood(binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    binomial = CondBinomial(Scope([0], [1]), n=1)

    p = random.random()
    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[binomial] = {"p": p}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    binomial = CondBinomial(Scope([0], [1]), n=1)

    p = random.random()
    cond_f = lambda data: {"p": p}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[binomial] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(binomial, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(binomial, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):

    n = random.randint(2, 10)
    p = random.random()

    torch_binomial = CondBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})
    node_binomial = BaseCondBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(node_binomial, data)
    log_probs_torch = log_likelihood(torch_binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    n = random.randint(2, 10)
    p = tl.tensor(random.random(), requires_grad=True)

    torch_binomial = CondBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs_torch = log_likelihood(torch_binomial, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_binomial.n.grad is None)
    tc.assertTrue(p.grad is not None)

def test_likelihood_p_0(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # p = 0
    binomial = CondBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 0.0})

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(binomial, data)
    log_probs = log_likelihood(binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets, atol=0.001, rtol=0.001))

def test_likelihood_p_1(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # p = 1
    binomial = CondBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[0.0], [1.0]])

    probs = likelihood(binomial, data)
    log_probs = log_likelihood(binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets, atol=0.001, rtol=0.001))

def test_likelihood_n_0(do_for_all_backends):

    # n = 0
    binomial = CondBinomial(Scope([0], [1]), 0, cond_f=lambda data: {"p": 0.5})

    data = tl.tensor([[0.0]])
    targets = tl.tensor([[1.0]])

    probs = likelihood(binomial, data)
    log_probs = log_likelihood(binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_marginalization(do_for_all_backends):

    binomial = CondBinomial(Scope([0], [1]), 5, cond_f=lambda data: {"p": 0.5})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1 (0 in log-space)
    probs = likelihood(binomial, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Binomial distribution: integers {0,...,n}

    binomial = CondBinomial(Scope([0], [1]), 2, cond_f=lambda data: {"p": 0.5})

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[-np.inf]]))
    tc.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[np.inf]]))

    # check valid integers inside valid range
    log_likelihood(
        binomial,
        tl_unsqueeze(tl.tensor(list(range(binomial.n + 1))), 1),
    )

    # check valid integers, but outside of valid range
    tc.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[-1]]))
    tc.assertRaises(
        ValueError,
        log_likelihood,
        binomial,
        tl.tensor([[float(binomial.n + 1)]]),
    )

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        binomial,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        binomial,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        binomial,
        tl.tensor(
            [
                [
                    np.nextafter(
                        tl.tensor(float(binomial.n)),
                        tl.tensor(float(binomial.n + 1)),
                    )
                ]
            ]
        ),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        binomial,
        tl.tensor([[np.nextafter(tl.tensor(float(binomial.n)), tl.tensor(0.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[0.5]]))
    tc.assertRaises(ValueError, log_likelihood, binomial, tl.tensor([[3.5]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    n = random.randint(2, 10)
    p = random.random()

    binomial = CondBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})


    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            binomial_updated = updateBackend(binomial)
            log_probs_updated = log_likelihood(binomial_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
