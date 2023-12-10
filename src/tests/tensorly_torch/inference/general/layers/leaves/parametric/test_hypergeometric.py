import unittest

import torch
import tensorly as tl
import numpy as np

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_hypergeometric import HypergeometricLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_hypergeometric import Hypergeometric
from spflow.torch.structure.general.layers.leaves.parametric.hypergeometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    layer = HypergeometricLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        N=[5, 7, 5],
        M=[2, 5, 2],
        n=[3, 2, 3],
    )

    nodes = [
        Hypergeometric(Scope([0]), N=5, M=2, n=3),
        Hypergeometric(Scope([1]), N=7, M=5, n=2),
        Hypergeometric(Scope([0]), N=5, M=2, n=3),
    ]

    dummy_data = tl.tensor([[2, 1], [0, 2], [1, 1]])

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    N = 15
    M = 10
    n = 10

    torch_hypergeometric = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=N, M=M, n=n)

    # create dummy input data (batch size x random variables)
    data = tl.tensor([[5, 6], [10, 5]])

    log_probs_torch = log_likelihood(torch_hypergeometric, data)

    # create dummy targets
    targets_torch = torch.ones(2, 2)
    targets_torch.requires_grad = True

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_hypergeometric.N.grad is None)
    tc.assertTrue(torch_hypergeometric.M.grad is None)
    tc.assertTrue(torch_hypergeometric.n.grad is None)

    # make sure distribution has no (learnable) parameters
    #tc.assertFalse(list(torch_hypergeometric.parameters()))

def test_likelihood_marginalization(do_for_all_backends):

    hypergeometric = HypergeometricLayer(scope=[Scope([0]), Scope([1])], N=5, M=3, n=4)
    data = tl.tensor([[float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(hypergeometric, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    layer = HypergeometricLayer(
        scope=[Scope([0]), Scope([1]), Scope([0])],
        N=[5, 7, 5],
        M=[2, 5, 2],
        n=[3, 2, 3],
    )
    dummy_data = tl.tensor([[2, 1], [0, 2], [1, 1]])

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
