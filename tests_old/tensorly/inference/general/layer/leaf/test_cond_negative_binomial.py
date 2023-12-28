import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.meta.dispatch import DispatchContext
from spflow.modules.module import log_likelihood, likelihood
from spflow.modules.layer import CondNegativeBinomialLayer
from spflow.structure.general.node.leaf.general_cond_negative_binomial import CondNegativeBinomial
from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_likelihood_no_p(do_for_all_backends):
    negative_binomial = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)
    tc.assertRaises(
        ValueError,
        log_likelihood,
        negative_binomial,
        tl.tensor([[0], [1]]),
    )


def test_likelihood_module_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    cond_f = lambda data: {"p": [1.0, 1.0]}

    negative_binomial = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]], dtype=tl.float32)
    targets = tl.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=tl.float32)

    probs = likelihood(negative_binomial, data)
    log_probs = log_likelihood(negative_binomial, data)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_likelihood_args_p(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    negative_binomial = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[negative_binomial] = {"p": [1.0, 1.0]}

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]])
    targets = tl.tensor([[1.0, 1.0], [0.0, 0.0]])

    probs = likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(negative_binomial, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_likelihood_args_cond_f(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    bernoulli = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2)

    cond_f = lambda data: {"p": tl.tensor([1.0, 1.0])}

    dispatch_ctx = DispatchContext()
    dispatch_ctx.args[bernoulli] = {"cond_f": cond_f}

    # create test inputs/outputs
    data = tl.tensor([[0], [1]])
    targets = tl.tensor([[1.0, 1.0], [0.0, 0.0]])

    probs = likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)
    log_probs = log_likelihood(bernoulli, data, dispatch_ctx=dispatch_ctx)

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.exp(log_probs)))
    tc.assertTrue(np.allclose(tle.toNumpy(probs), targets, atol=0.001, rtol=0.001))


def test_layer_likelihood(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    layer = CondNegativeBinomialLayer(
        scope=[Scope([0], [2]), Scope([1], [2]), Scope([0], [2])],
        n=[3, 2, 3],
        cond_f=lambda data: {"p": [0.2, 0.5, 0.9]},
    )

    nodes = [
        CondNegativeBinomial(Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.2}),
        CondNegativeBinomial(Scope([1], [2]), n=2, cond_f=lambda data: {"p": 0.5}),
        CondNegativeBinomial(Scope([0], [2]), n=3, cond_f=lambda data: {"p": 0.9}),
    ]

    dummy_data = tl.tensor([[3, 1], [1, 2], [0, 0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))


def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    n = [random.randint(2, 10), random.randint(2, 10)]
    p = tl.tensor([random.random(), random.random()], requires_grad=True)

    torch_negative_binomial = CondNegativeBinomialLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        n=n,
        cond_f=lambda data: {"p": p},
    )

    # create dummy input data (batch size x random variables)
    data = torch.cat(
        [torch.randint(1, n[0], (3, 1)), torch.randint(1, n[1], (3, 1))],
        dim=1,
    )

    log_probs_torch = log_likelihood(torch_negative_binomial, data)

    # create dummy targets
    targets_torch = tl.ones((3, 2))

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_negative_binomial.n.grad is None)
    tc.assertTrue(p.grad is not None)


def test_likelihood_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    negative_binomial = CondNegativeBinomialLayer(
        scope=[Scope([0], [2]), Scope([1], [2])],
        n=5,
        cond_f=lambda data: {"p": random.random()},
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(negative_binomial, data))

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor([1.0, 1.0])))


def test_support(do_for_all_backends):
    # TODO
    pass


def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    cond_f = lambda data: {"p": [1.0, 1.0]}

    negative_binomial = CondNegativeBinomialLayer(Scope([0], [1]), n=2, n_nodes=2, cond_f=cond_f)

    # create test inputs/outputs
    data = tl.tensor([[0.0], [1.0]], dtype=tl.float32)

    probs = likelihood(negative_binomial, data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(negative_binomial)
            probs_updated = likelihood(layer_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(probs), tle.toNumpy(probs_updated), atol=0.001, rtol=0.001))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
