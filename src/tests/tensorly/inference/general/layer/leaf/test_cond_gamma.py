import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_cond_gamma import CondGammaLayer
from spflow.tensorly.structure.general.node.leaf.general_cond_gamma import CondGamma
from spflow.torch.structure.general.layer.leaf.cond_gamma import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_likelihood_no_alpha(do_for_all_backends):

    gamma = CondGammaLayer(Scope([0], [1]), cond_f=lambda data: {"beta": [1.0, 1.0]}, n_nodes=2)
    tc.assertRaises(KeyError, log_likelihood, gamma, tl.tensor([[0], [1]]))

def test_likelihood_no_beta(do_for_all_backends):

    gamma = CondGammaLayer(
        Scope([0], [1]),
        cond_f=lambda data: {"alpha": [1.0, 1.0]},
        n_nodes=2,
    )
    tc.assertRaises(KeyError, log_likelihood, gamma, tl.tensor([[0], [1]]))

def test_likelihood_no_alpha_beta(do_for_all_backends):

    gamma = CondGammaLayer(Scope([0], [1]), n_nodes=2)
    tc.assertRaises(ValueError, log_likelihood, gamma, tl.tensor([[0], [1]]))

def test_likelihood_module_cond_f(do_for_all_backends):

    cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    gamma = CondGammaLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
    targets = tl.tensor([[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]])

    probs = likelihood(gamma, data)
    log_probs = log_likelihood(gamma, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args(do_for_all_backends):

    gamma = CondGammaLayer(Scope([0], [1]), n_nodes=2)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[gamma] = {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    # create test inputs/outputs
    data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
    targets = tl.tensor([[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]])

    probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_likelihood_args_cond_f(do_for_all_backends):

    gamma = CondGammaLayer(Scope([0], [1]), n_nodes=2)

    cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[gamma] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])
    targets = tl.tensor([[0.904837, 0.904837], [0.367879, 0.367879], [0.0497871, 0.0497871]])

    probs = likelihood(gamma, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(gamma, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), targets))

def test_layer_likelihood(do_for_all_backends):

    layer = CondGammaLayer(
        scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
        cond_f=lambda data: {
            "alpha": [0.2, 1.0, 2.3],
            "beta": [1.0, 0.3, 0.97],
        },
    )

    nodes = [
        CondGamma(Scope([0], [2]), cond_f=lambda data: {"alpha": 0.2, "beta": 1.0}),
        CondGamma(Scope([1], [2]), cond_f=lambda data: {"alpha": 1.0, "beta": 0.3}),
        CondGamma(
            Scope([0], [2]),
            cond_f=lambda data: {"alpha": 2.3, "beta": 0.97},
        ),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    alpha = tl.tensor(
        [random.randint(1, 5), random.randint(1, 3)],
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )
    beta = tl.tensor(
        [random.randint(1, 5), random.randint(2, 4)],
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )

    torch_gamma = CondGammaLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {"alpha": alpha, "beta": beta},
    )

    # create dummy input data (batch size x random variables)
    data = torch.rand(3, 2)

    log_probs_torch = log_likelihood(torch_gamma, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(alpha.grad is not None)
    tc.assertTrue(beta.grad is not None)

def test_likelihood_marginalization(do_for_all_backends):

    gamma = CondGammaLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {
            "alpha": random.random() + 1e-7,
            "beta": random.random() + 1e-7,
        },
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(gamma, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    gamma = CondGammaLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]])

    log_probs = log_likelihood(gamma, data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(gamma)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))

def test_change_dtype(do_for_all_backends):
    cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    layer = CondGammaLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)

def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    cond_f = lambda data: {"alpha": [1.0, 1.0], "beta": [1.0, 1.0]}

    layer = CondGammaLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    if do_for_all_backends == "numpy":
        tc.assertRaises(ValueError, layer.to_device, cuda)
        return
    tc.assertTrue(layer_ll.device.type == "cpu")
    layer.to_device(cuda)
    dummy_data = tl.tensor([[0.1, 0.1], [1.0, 1.0], [3.0, 3.0]], device=cuda)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.device.type == "cuda")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
