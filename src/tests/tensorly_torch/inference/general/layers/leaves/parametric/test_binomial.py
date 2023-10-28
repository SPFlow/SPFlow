import random
import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_binomial import BinomialLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_binomial import Binomial
from spflow.torch.structure.general.layers.leaves.parametric.binomial import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    layer = BinomialLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        n=[3, 2, 3],
        p=[0.2, 0.5, 0.9],
    )

    nodes = [
        Binomial(Scope([0]), n=3, p=0.2),
        Binomial(Scope([1]), n=2, p=0.5),
        Binomial(Scope([0]), n=3, p=0.9),
    ]

    dummy_data = tl.tensor([[3, 1], [1, 2], [0, 0]], dtype=tl.float64)

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    if do_for_all_backends == "numpy":
        return

    n = [4, 6]
    p = [random.random(), random.random()]

    binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=n, p=p)

    # create dummy input data (batch size x random variables)
    data = tl.tensor([[0, 5], [3, 2], [4, 1]], dtype=tl.float64)

    log_probs_torch = log_likelihood(binomial, data)

    # create dummy targets
    targets_torch = tl.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(binomial.n.grad is None)
    tc.assertTrue(binomial.p_aux.grad is not None)

    n_orig = binomial.n.detach().clone()
    p_aux_orig = binomial.p_aux.detach().clone()

    optimizer = torch.optim.SGD(binomial.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(np.allclose(n_orig, binomial.n))
    tc.assertTrue(np.allclose(tl_toNumpy(p_aux_orig - binomial.p_aux.grad), tl_toNumpy(binomial.p_aux)))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.equal(binomial.n, binomial.dist().total_count.long()))
    tc.assertTrue(np.allclose(tl_toNumpy(binomial.p), tl_toNumpy(binomial.dist().probs)))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=[5, 3], p=0.3)

    # create dummy data
    p_target = tl.tensor([0.8, 0.5], dtype=tl.float64)
    data = torch.distributions.Binomial(tl.tensor([5, 3]), p_target).sample((100000,))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(binomial.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(50):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(binomial, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(np.allclose(tl_toNumpy(binomial.p), tl_toNumpy(p_target), atol=1e-3, rtol=1e-3))

def test_likelihood_marginalization(do_for_all_backends):
    torch.set_default_dtype(torch.float64)

    binomial = BinomialLayer(scope=[Scope([0]), Scope([1])], n=5, p=random.random())
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(binomial, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float64)
    backends = ["numpy", "pytorch"]
    layer = BinomialLayer(
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
            tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
