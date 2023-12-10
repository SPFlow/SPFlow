import random
import unittest

import numpy as np
import torch
from packaging import version
import tensorly as tl

from spflow.tensorly.structure.spn import CondGamma
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.cond_gamma import CondGamma as BaseCondGamma
from spflow.torch.structure.general.nodes.leaves.parametric.cond_gamma import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):

    cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

    gamma = CondGamma(Scope([0], [1]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.1], [1.0], [3.0]])
    targets = tl.tensor([[0.904837], [0.367879], [0.0497871]])

    probs = likelihood(gamma, data)
    log_probs = log_likelihood(gamma, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    gamma = CondGamma(Scope([0], [1]))

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[gamma] = {"alpha": 1.0, "beta": 1.0}

    # create test inputs/outputs
    data = tl.tensor([[0.1], [1.0], [3.0]])
    targets = tl.tensor([[0.904837], [0.367879], [0.0497871]])

    probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    gamma = CondGamma(Scope([0], [1]))

    cond_f = lambda data: {"alpha": 1.0, "beta": 1.0}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[gamma] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0.1], [1.0], [3.0]])
    targets = tl.tensor([[0.904837], [0.367879], [0.0497871]])

    probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    alpha = tl.tensor(random.randint(1, 5), dtype=tl.float64)
    beta = tl.tensor(random.randint(1, 5), dtype=tl.float64)

    gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})
    node_gamma = BaseCondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(node_gamma, data)
    log_probs_torch = log_likelihood(gamma, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch), atol=0.001, rtol=0.001))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    alpha = tl.tensor(
        random.randint(1, 5),
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )
    beta = tl.tensor(
        random.randint(1, 5),
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )

    gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs_torch = log_likelihood(gamma, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(alpha.grad is not None)
    tc.assertTrue(beta.grad is not None)

def test_marginalization(do_for_all_backends):

    gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(gamma, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    # Support for Gamma distribution: floats (0,inf)

    # TODO:
    #   likelihood:     x=0 -> POS_EPS (?)
    #   log-likelihood: x=0 -> POS_EPS (?)

    gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": 1.0, "beta": 1.0})

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
    tc.assertTrue(np.allclose(probs, tl.exp(log_probs)))

    # check invalid float values (outside range)
    if version.parse(torch.__version__) < version.parse("1.12.0") or do_for_all_backends == "numpy":
        # edge case 0
        # scipy gamma distribution has no support for 0
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
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    alpha = tl.tensor(random.randint(1, 5), dtype=tl.float64)
    beta = tl.tensor(random.randint(1, 5), dtype=tl.float64)

    gamma = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(gamma, tl.tensor(data, dtype=tl.float64))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            gamma_updated = updateBackend(gamma)
            log_probs_updated = log_likelihood(gamma_updated, tl.tensor(data, dtype=tl.float64))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    alpha = tl.tensor(random.randint(1, 5), dtype=tl.float64)
    beta = tl.tensor(random.randint(1, 5), dtype=tl.float64)

    node = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})
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
    alpha = tl.tensor(random.randint(1, 5), dtype=tl.float64)
    beta = tl.tensor(random.randint(1, 5), dtype=tl.float64)

    node = CondGamma(Scope([0], [1]), cond_f=lambda data: {"alpha": alpha, "beta": beta})
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
