import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import CondBernoulli
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.cond_bernoulli import CondBernoulli as BaseCondBernoulli
from spflow.torch.structure.general.nodes.leaves.parametric.cond_bernoulli import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):

    p = random.random()
    cond_f = lambda data: {"p": p}

    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    bernoulli = CondBernoulli(Scope([0], [1]))

    p = random.random()
    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[bernoulli] = {"p": p}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    bernoulli = CondBernoulli(Scope([0], [1]))

    p = random.random()
    cond_f = lambda data: {"p": p}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0 - p], [p]])

    probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):

    p = np.array(0.5)

    torch_bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": tl.tensor(p)})
    node_bernoulli = BaseCondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))

    log_probs = log_likelihood(node_bernoulli, data)
    log_probs_torch = log_likelihood(torch_bernoulli, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    p = tl.tensor(0.5, requires_grad=True)

    torch_bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))

    log_probs_torch = log_likelihood(torch_bernoulli, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(p.grad is not None)

def test_likelihood_p_0(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # p = 0
    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 0.0})

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0], [0.0]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets, atol=0.001, rtol=0.001))

def test_likelihood_p_1(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # p = 1
    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": 1.0})

    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[0.0], [1.0]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets, atol=0.001, rtol=0.001))

def test_likelihood_marginalization(do_for_all_backends):

    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": random.random()})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Bernoulli distribution: integers {0,1}

    p = random.random()
    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": p})

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

def test_update_backend_1(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    p = np.array(0.5)

    bernoulli = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": tl.tensor(p)})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 1))
    log_probs = log_likelihood(bernoulli, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            binomial_updated = updateBackend(bernoulli)
            log_probs_updated = log_likelihood(binomial_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    p = np.array(0.5)

    node = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": tl.tensor(p)})
    dummy_data = tl.tensor(np.random.randint(0, 2, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    p = np.array(0.5)

    node = CondBernoulli(Scope([0], [1]), cond_f=lambda data: {"p": tl.tensor(p)})
    dummy_data = tl.tensor(np.random.randint(0, 2, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.randint(0, 2, (3, 1)), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")







if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
