import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_gamma import GammaLayer
from spflow.tensorly.structure.general.node.leaf.general_gamma import Gamma
from spflow.torch.structure.general.layer.leaf.gamma import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.0, 2.3],
        beta=[1.0, 0.3, 0.97],
    )

    nodes = [
        Gamma(Scope([0]), alpha=0.2, beta=1.0),
        Gamma(Scope([1]), alpha=1.0, beta=0.3),
        Gamma(Scope([0]), alpha=2.3, beta=0.97),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    alpha = [random.randint(1, 5), random.randint(1, 3)]
    beta = [random.randint(1, 5), random.randint(2, 4)]

    torch_gamma = GammaLayer(scope=[Scope([0]), Scope([1])], alpha=alpha, beta=beta)

    # create dummy input data (batch size x random variables)
    data = torch.rand(3, 2)

    log_probs_torch = log_likelihood(torch_gamma, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_gamma.alpha_aux.grad is not None)
    tc.assertTrue(torch_gamma.beta_aux.grad is not None)

    alpha_aux_orig = torch_gamma.alpha_aux.detach().clone()
    beta_aux_orig = torch_gamma.beta_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        torch.allclose(
            alpha_aux_orig - torch_gamma.alpha_aux.grad,
            torch_gamma.alpha_aux,
        )
    )
    tc.assertTrue(torch.allclose(beta_aux_orig - torch_gamma.beta_aux.grad, torch_gamma.beta_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_gamma.alpha, torch_gamma.dist().concentration))
    tc.assertTrue(torch.allclose(torch_gamma.beta, torch_gamma.dist().rate))

def test_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_gamma = GammaLayer(scope=[Scope([0]), Scope([1])], alpha=[1.0, 1.2], beta=[2.0, 1.8])

    torch.manual_seed(0)

    # create dummy data
    data = torch.distributions.Gamma(concentration=2.0, rate=1.0).sample((100000, 2))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_gamma.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_gamma, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(torch.allclose(torch_gamma.alpha, tl.tensor([2.0, 2.0], dtype=tl.float32), atol=1e-2, rtol=0.3))
    tc.assertTrue(torch.allclose(torch_gamma.beta, tl.tensor([1.0, 1.0], dtype=tl.float32), atol=1e-2, rtol=0.3))

def test_likelihood_marginalization(do_for_all_backends):

    gamma = GammaLayer(
        scope=[Scope([0]), Scope([1])],
        alpha=random.random() + 1e-7,
        beta=random.random() + 1e-7,
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(gamma, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = GammaLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        alpha=[0.2, 1.0, 2.3],
        beta=[1.0, 0.3, 0.97],
    )

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
    torch.set_default_dtype(torch.float32)
    unittest.main()
