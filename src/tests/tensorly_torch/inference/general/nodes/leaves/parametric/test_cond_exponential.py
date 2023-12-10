import random
import unittest

import numpy as np
import torch
from packaging import version
import tensorly as tl

from spflow.tensorly.structure.spn import CondExponential
from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.cond_exponential import CondExponential as BaseCondExponential
from spflow.torch.structure.general.nodes.leaves.parametric.cond_exponential import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_module_cond_f(do_for_all_backends):

    cond_f = lambda data: {"l": 0.5}

    exponential = CondExponential(Scope([0], [1]), cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394], [0.0410425]])

    probs = likelihood(exponential, data)
    log_probs = log_likelihood(exponential, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_p(do_for_all_backends):

    exponential = CondExponential(Scope([0], [1]))

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[exponential] = {"l": 0.5}

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394], [0.0410425]])

    probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    exponential = CondExponential(Scope([0], [1]))

    cond_f = lambda data: {"l": 0.5}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[exponential] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394], [0.0410425]])

    probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_inference(do_for_all_backends):

    l = random.random() + 1e-7  # small offset to avoid zero

    torch_exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})
    node_exponential = BaseCondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(node_exponential, data)
    log_probs_torch = log_likelihood(torch_exponential, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    l = tl.tensor(random.random(), requires_grad=True)

    torch_exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})

    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs_torch = log_likelihood(torch_exponential, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 1)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(l.grad is not None)

def test_likelihood_marginalization(do_for_all_backends):

    exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.0})
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(exponential, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Exponential distribution: floats [0,inf) (note: 0 excluded in pytorch support)

    exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": 1.5})

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[-float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[float("inf")]]),
    )

    # check valid float values (within range)
    log_likelihood(
        exponential,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(1.0))]]),
    )
    log_likelihood(exponential, tl.tensor([[10.5]]))

    # check invalid float values (outside range)
    tc.assertRaises(
        ValueError,
        log_likelihood,
        exponential,
        tl.tensor([[np.nextafter(tl.tensor(0.0), tl.tensor(-1.0))]]),
    )

    if version.parse(torch.__version__) < version.parse("1.11.0"):
        # edge case 0 (part of the support in scipy, but NOT pytorch)
        tc.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[0.0]]))
    else:
        # edge case 0
        log_likelihood(exponential, tl.tensor([[0.0]]))

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    l = random.random() + 1e-7  # small offset to avoid zero

    exponential = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})


    # create dummy input data (batch size x random variables)
    data = np.random.rand(3, 1)

    log_probs = log_likelihood(exponential, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            exponential_updated = updateBackend(exponential)
            log_probs_updated = log_likelihood(exponential_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    l = random.random() + 1e-7  # small offset to avoid zero

    node = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})
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
    l = random.random() + 1e-7  # small offset to avoid zero

    node = CondExponential(Scope([0], [1]), cond_f=lambda data: {"l": l})
    dummy_data = tl.tensor(np.random.randint(0, 2, (3, 1)), dtype=tl.float32)
    layer_ll = log_likelihood(node, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, node.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    node.to_device(cuda)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], device=cuda)
    layer_ll = log_likelihood(node, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
