import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood, likelihood
from spflow.structure.general.layer.leaf import CondBernoulliLayer
from spflow.structure.general.node.leaf.general_cond_bernoulli import CondBernoulli
from spflow.torch.structure.general.layer.leaf.cond_bernoulli import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_likelihood_no_p(do_for_all_backends):
    bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)
    tc.assertRaises(ValueError, log_likelihood, bernoulli, tl.tensor([[0], [1]]))


def test_likelihood_module_cond_f(do_for_all_backends):
    cond_f = lambda data: {"p": [0.8, 0.5]}

    bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

    probs = likelihood(bernoulli, data)
    log_probs = log_likelihood(bernoulli, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_likelihood_args_p(do_for_all_backends):
    bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[bernoulli] = {"p": [0.8, 0.5]}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

    probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_likelihood_args_cond_f(do_for_all_backends):
    bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2)

    cond_f = lambda data: {"p": tl.tensor([0.8, 0.5])}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[0.2, 0.5], [0.8, 0.5]])

    probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets))


def test_layer_likelihood(do_for_all_backends):
    layer = CondBernoulliLayer(
        scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
        cond_f=lambda data: {"p": [0.2, 0.5, 0.9]},
    )

    nodes = [
        CondBernoulli(Scope([0], [2]), cond_f=lambda data: {"p": 0.2}),
        CondBernoulli(Scope([1], [2]), cond_f=lambda data: {"p": 0.5}),
        CondBernoulli(Scope([0], [2]), cond_f=lambda data: {"p": 0.9}),
    ]

    dummy_data = tl.tensor([[1, 0], [0, 0], [1, 1]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))


def test_layer_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    p = tl.tensor([random.random(), random.random()], requires_grad=True)

    torch_bernoulli = CondBernoulliLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {"p": p},
    )

    # create dummy input data (batch size x random variables)
    data = np.random.randint(0, 2, (3, 2))

    log_probs_torch = log_likelihood(torch_bernoulli, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(p.grad is not None)


def test_likelihood_marginalization(do_for_all_backends):
    bernoulli = CondBernoulliLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        cond_f=lambda data: {"p": random.random()},
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(bernoulli, data))

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor([1.0, 1.0])))


def test_support(do_for_all_backends):
    # TODO
    pass


def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    cond_f = lambda data: {"p": [0.8, 0.5]}

    bernoulli = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    dummy_data = tl.tensor([[0], [1]])
    log_probs = log_likelihood(bernoulli, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(bernoulli)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(log_probs), tle.toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    cond_f = lambda data: {"p": [0.8, 0.5]}

    layer = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[0], [1]], dtype=tl.float32)
    layer_ll = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll.dtype == tl.float32)
    layer.to_dtype(tl.float64)
    dummy_data = tl.tensor([[0], [1]], dtype=tl.float64)
    layer_ll_up = log_likelihood(layer, dummy_data)
    tc.assertTrue(layer_ll_up.dtype == tl.float64)


def test_change_device(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    cuda = torch.device("cuda")
    cond_f = lambda data: {"p": [0.8, 0.5]}

    layer = CondBernoulliLayer(Scope([0], [1]), n_nodes=2, cond_f=cond_f)
    dummy_data = tl.tensor([[0], [1]], dtype=tl.float32)
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
