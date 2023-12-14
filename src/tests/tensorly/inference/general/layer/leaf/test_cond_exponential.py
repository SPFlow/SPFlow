import random
import unittest

import torch
import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_cond_exponential import CondExponentialLayer
from spflow.tensorly.structure.general.node.leaf.general_cond_exponential import CondExponential
from spflow.torch.structure.general.layer.leaf.cond_exponential import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_no_l(do_for_all_backends):

    exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)
    tc.assertRaises(ValueError, log_likelihood, exponential, tl.tensor([[0], [1]]))

def test_likelihood_module_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    cond_f = lambda data: {"l": [0.5, 1.0]}

    exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

    probs = likelihood(exponential, data)
    log_probs = log_likelihood(exponential, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_l(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[exponential] = {"l": [0.5, 1.0]}

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

    probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2)

    cond_f = lambda data: {"l": tl.tensor([0.5, 1.0])}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[exponential] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    targets = tl.tensor([[0.18394, 0.135335], [0.0410425, 0.00673795]])

    probs = likelihood(exponential, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(exponential, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_layer_likelihood(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    layer = CondExponentialLayer(
        scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
        cond_f=lambda data: {"l": [0.2, 1.0, 2.3]},
    )

    nodes = [
        CondExponential(Scope([0], [2]), cond_f=lambda data: {"l": 0.2}),
        CondExponential(Scope([1], [2]), cond_f=lambda data: {"l": 1.0}),
        CondExponential(Scope([0], [2]), cond_f=lambda data: {"l": 2.3}),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    l = tl.tensor([random.random(), random.random()], requires_grad=True)

    torch_exponential = CondExponentialLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {"l": l},
    )

    # create dummy input data (batch size x random variables)
    data = tl.random.random_tensor((3, 2))

    log_probs_torch = log_likelihood(torch_exponential, data)

    # create dummy targets
    targets_torch = tl.ones((3, 2))

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(l.grad is not None)

def test_likelihood_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    exponential = CondExponentialLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {"l": random.random() + 1e-7},
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(exponential, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    cond_f = lambda data: {"l": [0.5, 1.0]}

    exponential = CondExponentialLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[2], [5]])
    log_probs = log_likelihood(exponential, data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(exponential)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    cond_f = lambda data: {"l": [0.5, 1.0]}

    layer = CondExponentialLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    cond_f = lambda data: {"l": [0.5, 1.0]}

    layer = CondExponentialLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]])
    layer_ll = log_likelihood(layer, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer.to_device(cuda)
    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]], device=cuda)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
