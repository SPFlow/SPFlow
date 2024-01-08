import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.modules.module import log_likelihood
from spflow.structure.general.layer.leaf import NegativeBinomialLayer
from spflow.modules.node import NegativeBinomial
from spflow.torch.structure.general.layer.leaf.negative_binomial import updateBackend
from spflow.utils import Tensor
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_layer_likelihood(do_for_all_backends):
    layer = NegativeBinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[3, 2, 3],
        p=[0.2, 0.5, 0.9],
    )

    nodes = [
        NegativeBinomial(Scope([0]), n=3, p=0.2),
        NegativeBinomial(Scope([1]), n=2, p=0.5),
        NegativeBinomial(Scope([0]), n=3, p=0.9),
    ]

    dummy_data = tl.tensor([[3, 1], [1, 2], [0, 0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))


def test_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    n = [random.randint(2, 10), random.randint(2, 10)]
    p = [random.random(), random.random()]

    torch_negative_binomial = NegativeBinomialLayer(scope=[Scope([0]), Scope([1])], n=n, p=p)

    # create dummy input data (batch size x random variables)
    data = torch.cat(
        [torch.randint(1, n[0], (3, 1)), torch.randint(1, n[1], (3, 1))],
        dim=1,
    )

    log_probs_torch = log_likelihood(torch_negative_binomial, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_negative_binomial.n.grad is None)
    tc.assertTrue(torch_negative_binomial.p_aux.grad is not None)

    n_orig = torch_negative_binomial.n.detach().clone()
    p_aux_orig = torch_negative_binomial.p_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(n_orig, torch_negative_binomial.n))
    tc.assertTrue(
        torch.allclose(
            p_aux_orig - torch_negative_binomial.p_aux.grad,
            torch_negative_binomial.p_aux,
        )
    )


def test_gradient_optimization(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    torch_negative_binomial = NegativeBinomialLayer(scope=[Scope([0]), Scope([1])], n=5, p=0.3)

    # create dummy data
    p_target = tl.tensor([0.8, 0.8], dtype=tl.float32)
    data = torch.distributions.NegativeBinomial(5, 1 - p_target).sample((100000,))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_negative_binomial.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(40):
        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_negative_binomial, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_negative_binomial.p, p_target, atol=1e-3, rtol=1e-3))


def test_likelihood_marginalization(do_for_all_backends):
    negative_binomial = NegativeBinomialLayer(scope=[Scope([0]), Scope([1])], n=5, p=random.random())
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
    layer = NegativeBinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[3, 2, 3],
        p=[0.2, 0.5, 0.9],
    )

    dummy_data = tl.tensor([[3, 1], [1, 2], [0, 0]])

    layer_ll = log_likelihood(layer, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(log_probs_updated)))


def test_change_dtype(do_for_all_backends):
    layer = NegativeBinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[3, 2, 3],
        p=[0.2, 0.5, 0.9],
    )
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
    layer = NegativeBinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[3, 2, 3],
        p=[0.2, 0.5, 0.9],
    )
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
