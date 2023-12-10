import random
import unittest

import torch
import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_log_normal import LogNormalLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_log_normal import LogNormal
from spflow.torch.structure.general.layers.leaves.parametric.log_normal import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):

    layer = LogNormalLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        mean=[0.2, 1.0, 2.3],
        std=[1.0, 0.3, 0.97],
    )

    nodes = [
        LogNormal(Scope([0]), mean=0.2, std=1.0),
        LogNormal(Scope([1]), mean=1.0, std=0.3),
        LogNormal(Scope([0]), mean=2.3, std=0.97),
    ]

    dummy_data = tl.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    mean = [random.random(), random.random()]
    std = [
        random.random() + 1e-8,
        random.random() + 1e-8,
    ]  # offset by small number to avoid zero

    torch_log_normal = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=mean, std=std)

    # create dummy input data (batch size x random variables)
    data = torch.rand(3, 2)

    log_probs_torch = log_likelihood(torch_log_normal, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_log_normal.mean.grad is not None)
    tc.assertTrue(torch_log_normal.std_aux.grad is not None)

    mean_orig = torch_log_normal.mean.detach().clone()
    std_aux_orig = torch_log_normal.std_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(mean_orig - torch_log_normal.mean.grad, torch_log_normal.mean))
    tc.assertTrue(
        torch.allclose(
            std_aux_orig - torch_log_normal.std_aux.grad,
            torch_log_normal.std_aux,
        )
    )

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_log_normal.mean, torch_log_normal.dist().loc))
    tc.assertTrue(torch.allclose(torch_log_normal.std, torch_log_normal.dist().scale))

def test_gradient_optimization(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    if do_for_all_backends == "numpy":
        return

    # initialize distribution
    torch_log_normal = LogNormalLayer(scope=[Scope([0]), Scope([1])], mean=[1.1, 0.8], std=[2.0, 1.5])

    torch.manual_seed(10)

    # create dummy data
    data = torch.distributions.LogNormal(0.0, 1.0).sample((100000, 2))

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_log_normal.parameters(), lr=0.5, momentum=0.5)

    # perform optimization (possibly overfitting)
    for i in range(20):

        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_log_normal, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(
            torch_log_normal.mean,
            tl.tensor([0.0, 0.0], dtype=tl.float32),
            atol=1e-2,
            rtol=0.2,
        )
    )
    tc.assertTrue(
        torch.allclose(
            torch_log_normal.std,
            tl.tensor([1.0, 1.0], dtype=tl.float32),
            atol=1e-2,
            rtol=0.2,
        )
    )

def test_likelihood_marginalization(do_for_all_backends):

    log_normal = LogNormalLayer(
        scope=[Scope([0]), Scope([1])],
        mean=random.random(),
        std=random.random() + 1e-7,
    )
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(log_normal, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = LogNormalLayer(
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
