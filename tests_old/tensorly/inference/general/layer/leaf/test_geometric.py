import random
import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.modules.module import log_likelihood
from spflow.modules.layer import GeometricLayer
from spflow.modules.node import Geometric
from spflow.torch.structure.general.layer.leaf.geometric import updateBackend
from spflow.tensor import ops as tle

tc = unittest.TestCase()


def test_layer_likelihood(do_for_all_backends):
    layer = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 1.0, 0.3])

    nodes = [
        Geometric(Scope([0]), p=0.2),
        Geometric(Scope([1]), p=1.0),
        Geometric(Scope([0]), p=0.3),
    ]

    dummy_data = tl.tensor([[4, 1], [3, 7], [1, 1]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(nodes_ll)))


def test_gradient_computation(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    p = [random.random(), random.random()]

    torch_geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=p)

    # create dummy input data (batch size x random variables)
    data = torch.randint(1, 10, (3, 2))

    log_probs_torch = log_likelihood(torch_geometric, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_geometric.p_aux.grad is not None)

    p_aux_orig = torch_geometric.p_aux.detach().clone()

    optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(torch.allclose(p_aux_orig - torch_geometric.p_aux.grad, torch_geometric.p_aux))

    # verify that distribution parameters match parameters
    tc.assertTrue(torch.allclose(torch_geometric.p, torch_geometric.dist().probs))


def test_gradient_optimization(do_for_all_backends):
    if do_for_all_backends == "numpy":
        return

    torch.manual_seed(0)

    # initialize distribution
    torch_geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=[0.3, 0.4])

    # create dummy data
    p_target = 0.8
    data = torch.distributions.Geometric(p_target).sample((100000, 2)) + 1

    # initialize gradient optimizer
    optimizer = torch.optim.SGD(torch_geometric.parameters(), lr=0.9, momentum=0.6)

    # perform optimization (possibly overfitting)
    for i in range(40):
        # clear gradients
        optimizer.zero_grad()

        # compute negative log-likelihood
        nll = -log_likelihood(torch_geometric, data).mean()
        nll.backward()

        # update parameters
        optimizer.step()

    tc.assertTrue(
        torch.allclose(torch_geometric.p, tl.tensor(p_target, dtype=tl.float32), atol=1e-3, rtol=1e-3)
    )


def test_likelihood_marginalization(do_for_all_backends):
    geometric = GeometricLayer(scope=[Scope([0]), Scope([1])], p=random.random() + 1e-7)
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(geometric, data))

    tc.assertTrue(np.allclose(tle.toNumpy(probs), tl.tensor([1.0, 1.0])))


def test_support(do_for_all_backends):
    # TODO
    pass


def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float32)
    backends = ["numpy", "pytorch"]
    layer = GeometricLayer(scope=[Scope([0]), Scope([1]), Scope([0])], p=[0.2, 1.0, 0.3])

    dummy_data = tl.tensor([[4, 1], [3, 7], [1, 1]])

    layer_ll = log_likelihood(layer, dummy_data)

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tle.toNumpy(layer_ll), tle.toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    unittest.main()
