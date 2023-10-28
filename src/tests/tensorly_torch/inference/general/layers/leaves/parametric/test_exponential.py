import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_exponential import ExponentialLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_exponential import Exponential
from spflow.torch.structure.general.layers.leaves.parametric.exponential import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

    nodes = [
        Exponential(Scope([0]), l=0.2),
        Exponential(Scope([1]), l=1.0),
        Exponential(Scope([0]), l=2.3),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    l = [random.random(), random.random()]

    torch_exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=l)

    # create dummy input data (batch size x random variables)
    data = torch.rand(3, 2)

    log_probs_torch = log_likelihood(torch_exponential, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_exponential.l_aux.grad is not None)

    l_aux_orig = torch_exponential.l_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        torch.allclose(
            l_aux_orig - torch_exponential.l_aux.grad,
            torch_exponential.l_aux,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_exponential.l, torch_exponential.dist().rate))

def test_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=[0.5, 0.7])

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Exponential(rate=1.5).sample((100000, 2))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_exponential.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_exponential, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(
            torch_exponential.l,
            tl.tensor([1.5, 1.5], dtype=tl.float64),
            atol=1e-3,
            rtol=0.3,
        )
    )

def test_likelihood_marginalization(do_for_all_backends):

    exponential = ExponentialLayer(scope=[Scope([0]), Scope([1])], l=random.random() + 1e-7)
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(exponential, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float64)
    backends = ["numpy", "pytorch"]
    layer = ExponentialLayer(scope=[Scope([0]), Scope([1]), Scope([0])], l=[0.2, 1.0, 2.3])

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

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
