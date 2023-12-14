import random
import unittest

import torch
import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_gaussian import GaussianLayer
from spflow.tensorly.structure.general.node.leaf.general_gaussian import Gaussian
from spflow.torch.structure.general.layer.leaf.gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = GaussianLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 1.0, 2.3],
        std=[1.0, 0.3, 0.97],
    )

    nodes = [
        Gaussian(Scope([0]), mean=0.2, std=1.0),
        Gaussian(Scope([1]), mean=1.0, std=0.3),
        Gaussian(Scope([0]), mean=2.3, std=0.97),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    mean = [random.random(), random.random()]
    std = [
        random.random() + 1e-8,
        random.random() + 1e-8,
    ]  # offset by small number to avoid zero

    torch_gaussian = GaussianLayer(scope=[Scope([0]), Scope([1])], mean=mean, std=std)

    # create dummy input data (batch size x random variables)
    data = torch.randn(3, 2)

    log_probs_torch = log_likelihood(torch_gaussian, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_gaussian.mean.grad is not None)
    tc.assertTrue(torch_gaussian.std_aux.grad is not None)

    mean_orig = torch_gaussian.mean.detach().clone()
    std_aux_orig = torch_gaussian.std_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(mean_orig - torch_gaussian.mean.grad, torch_gaussian.mean))
    tc.assertTrue(
        torch.allclose(
            std_aux_orig - torch_gaussian.std_aux.grad,
            torch_gaussian.std_aux,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_gaussian.mean, torch_gaussian.dist().mean))
    tc.assertTrue(torch.allclose(torch_gaussian.std, torch_gaussian.dist().stddev))

def test_gradient_optimization(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_gaussian = GaussianLayer(scope=[Scope([0]), Scope([1])], mean=[1.0, 1.1], std=[2.0, 1.9])

    torch.manual_seed(0)

    # create dummy data (unit variance Gaussian)
    data = torch.randn((100000, 2))
    data = (data - data.mean()) / data.std()

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=0.5)

    # perform optimization (possibly overfitting)
    for i in range(40):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_gaussian, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(
            torch_gaussian.mean,
            tl.tensor([0.0, 0.0], dtype=tl.float32),
            atol=1e-2,
            rtol=1e-2,
        )
    )
    tc.assertTrue(
        torch.allclose(
            torch_gaussian.std,
            tl.tensor([1.0, 1.0], dtype=tl.float32),
            atol=1e-2,
            rtol=1e-2,
        )
    )

def test_likelihood_marginalization(do_for_all_backends):

    gaussian = GaussianLayer(
        scope=[Scope([0]), Scope([1])],
        mean=random.random(),
        std=random.random() + 1e-7,
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(gaussian, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = GaussianLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 1.0, 2.3],
        std=[1.0, 0.3, 0.97],
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
