import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood, likelihood
from spflow.structure.spn import CondNegativeBinomial
from spflow.base.structure.general.node.leaf.cond_negative_binomial import (
    CondNegativeBinomial as BaseCondNegativeBinomial,
)
from spflow.torch.structure.general.node.leaf.cond_negative_binomial import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_likelihood_module_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    cond_f = lambda data: {"p": 1.0}

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(negative_binomial, data)
    log_probs = log_likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_likelihood_args_p(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[negative_binomial] = {"p": 1.0}

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_likelihood_args_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), n=1)

    cond_f = lambda data: {"p": 1.0}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[negative_binomial] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_inference(do_for_all_backends):
    n = random.randint(2, 10)
    p = random.random()

    torch_negative_binomial = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})
    node_negative_binomial = BaseCondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(node_negative_binomial, data)
    log_probs_torch = log_likelihood(torch_negative_binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tle.toNumpy(log_probs_torch)))


def test_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    n = random.randint(2, 10)
    p = tl.tensor(random.random(), requires_grad=True)

    torch_negative_binomial = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs_torch = log_likelihood(torch_negative_binomial, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_negative_binomial.n.grad is None)
    tc.assertTrue(p.grad is not None)


def test_likelihood_p_1(do_for_all_backends):
    # p = 1
    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 1.0})

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(negative_binomial, data)
    log_probs = log_likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_likelihood_n_float(do_for_all_backends):
    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 1, cond_f=lambda data: {"p": 0.5})
    tc.assertRaises(Exception, likelihood, negative_binomial, 0.5)


def test_likelihood_marginalization(do_for_all_backends):
    negative_binomial = CondNegativeBinomial(Scope([0], [1]), 20, cond_f=lambda data: {"p": 0.3})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor(1.0)))


def test_support(do_for_all_backends):
    # Support for Negative Binomial distribution: integers N U {0}

    n = 20
    p = 0.3

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

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

    negative_binomial = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(1, n, (3, 1))

    log_probs = log_likelihood(negative_binomial, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            negative_binomial_updated = updateBackend(negative_binomial)
            log_probs_updated = log_likelihood(negative_binomial_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(log_probs), tle.toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    n = random.randint(2, 10)
    p = random.random()

    node = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})
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

    node = CondNegativeBinomial(Scope([0], [1]), n, cond_f=lambda data: {"p": p})
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
