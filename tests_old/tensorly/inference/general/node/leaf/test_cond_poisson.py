import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.structure.spn import CondPoisson
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood, likelihood
from spflow.base.structure.general.node.leaf.cond_poisson import CondPoisson as BaseCondPoisson
from spflow.torch.structure.general.node.leaf.cond_poisson import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_likelihood_module_cond_f(do_for_all_backends):
    cond_f = lambda data: {"l": 1.0}

    poisson = CondPoisson(Scope([0], [1]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0], [2], [5]])
    targets = tl.tensor([[0.367879], [0.18394], [0.00306566]])

    probs = likelihood(poisson, data)
    log_probs = log_likelihood(poisson, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_likelihood_args_p(do_for_all_backends):
    poisson = CondPoisson(Scope([0], [1]))

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[poisson] = {"l": 1.0}

    # create test inputs/outputs
    data = tl.tensor([[0], [2], [5]])
    targets = tl.tensor([[0.367879], [0.18394], [0.00306566]])

    probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_likelihood_args_cond_f(do_for_all_backends):
    poisson = CondPoisson(Scope([0], [1]))

    cond_f = lambda data: {"l": 1.0}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[poisson] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0], [2], [5]])
    targets = tl.tensor([[0.367879], [0.18394], [0.00306566]])

    probs = likelihood(poisson, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(poisson, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_inference(do_for_all_backends):
    l = random.randint(1, 10)

    torch_poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})
    node_poisson = BaseCondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs = log_likelihood(node_poisson, data)
    log_probs_torch = log_likelihood(torch_poisson, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tle.toNumpy(log_probs_torch)))


def test_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    l = tl.tensor(
        random.randint(1, 10),
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )

    torch_poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs_torch = log_likelihood(torch_poisson, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(l.grad is not None)


def test_likelihood_marginalization(do_for_all_backends):
    poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(poisson, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor(1.0)))


def test_support(do_for_all_backends):
    # Support for Poisson distribution: integers N U {0}

    l = random.random()

    poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # check infinite values
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-float("inf")]]))
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[float("inf")]]))

    # check valid integers, but outside of valid range
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[-1]]))

    # check valid integers within valid range
    log_likelihood(poisson, tl.tensor([[0]]))
    log_likelihood(poisson, tl.tensor([[100]]))

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        poisson,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        poisson,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, poisson, tl.tensor([[10.1]]))


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.randint(1, 10)

    poisson = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 10, (3, 1))

    log_probs = log_likelihood(poisson, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            poisson_updated = updateBackend(poisson)
            log_probs_updated = log_likelihood(poisson_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(log_probs), tle.toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    l = random.randint(1, 10)

    node = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    node.to_dtype(tl.float64)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float64)
    layer_ll_up = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    l = random.randint(1, 10)

    node = CondPoisson(Scope([0], [1]), cond_f=lambda data: {"l": l})
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor(np.random.randint(0, 10, (3, 1)), device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()