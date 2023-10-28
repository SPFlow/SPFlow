import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.tensorly.structure.spn import Hypergeometric
from spflow.meta.data import Scope
from spflow.tensorly.inference import likelihood, log_likelihood
from spflow.base.structure.general.nodes.leaves.parametric.hypergeometric import Hypergeometric as BaseHypergeometric
from spflow.torch.structure.general.nodes.leaves.parametric.hypergeometric import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_inference(do_for_all_backends):

    N = 15
    M = 10
    n = 10

    torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)
    node_hypergeometric = BaseHypergeometric(Scope([0]), N, M, n)

    # create dummy input data (batch size x random variables)
    data = np.array([[5], [10]])

    log_probs = log_likelihood(node_hypergeometric, data)
    log_probs_torch = log_likelihood(torch_hypergeometric, tl.tensor(data))

    # TODO: support is handled differently (in log space): -inf for torch and np.finfo().min for numpy (decide how to handle)
    log_probs[log_probs == np.finfo(log_probs.dtype).min] = -np.inf

    # make sure that probabilities match python backend probabilities
    tc.assertTrue(np.allclose(log_probs, tl_toNumpy(log_probs_torch)))

def test_gradient_computation(do_for_all_backends):

    if do_for_all_backends == "numpy":
        return

    N = 15
    M = 10
    n = 10

    torch_hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    # create dummy input data (batch size x random variables)
    data = np.array([[5], [10]])

    log_probs_torch = log_likelihood(torch_hypergeometric, tl.tensor(data))

    # create dummy targets
    targets_torch = torch.ones(2, 1)
    targets_torch.requires_grad = True

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(torch_hypergeometric.N.grad is None)
    tc.assertTrue(torch_hypergeometric.M.grad is None)
    tc.assertTrue(torch_hypergeometric.n.grad is None)

    # make sure distribution has no (learnable) parameters
    #tc.assertFalse(list(torch_hypergeometric.parameters()))

def test_likelihood_marginalization(do_for_all_backends):

    hypergeometric = Hypergeometric(Scope([0]), 15, 10, 10)
    data = tl.tensor([[float("nan")]])

    # should not raise and error and should return 1
    probs = likelihood(hypergeometric, data)

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor(1.0)))

def test_support(do_for_all_backends):

    # Support for Hypergeometric distribution: integers {max(0,n+M-N),...,min(n,M)}

    # case n+M-N > 0
    N = 15
    M = 10
    n = 10

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    # check infinite values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[-float("inf")]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[float("inf")]]),
    )

    # check valid integers inside valid range
    data = tl.tensor([[max(0, n + M - N)], [min(n, M)]])

    probs = likelihood(hypergeometric, data)
    log_probs = log_likelihood(hypergeometric, data)

    tc.assertTrue(all(probs != 0))
    tc.assertTrue(np.allclose(probs, tl.exp(log_probs)))

    # check valid integers, but outside of valid range
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[max(0, n + M - N) - 1]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[min(n, M) + 1]]),
    )

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor(
            [
                [
                    np.nextafter(
                        tl.tensor(float(max(0, n + M - N))),
                        tl.tensor(100),
                    )
                ]
            ]
        ),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[np.nextafter(tl.tensor(float(max(n, M))), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, hypergeometric, tl.tensor([[5.5]]))

    # case n+M-N
    N = 25

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    # check valid integers within valid range
    data = tl.tensor([[max(0, n + M - N)], [min(n, M)]])

    probs = likelihood(hypergeometric, data)
    log_probs = log_likelihood(hypergeometric, data)

    tc.assertTrue(all(probs != 0))
    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.exp(log_probs)))

    # check valid integers, but outside of valid range
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[max(0, n + M - N) - 1]]),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[min(n, M) + 1]]),
    )

    # check invalid float values
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor(
            [
                [
                    np.nextafter(
                        tl.tensor(float(max(0, n + M - N))),
                        tl.tensor(100),
                    )
                ]
            ]
        ),
    )
    tc.assertRaises(
        ValueError,
        log_likelihood,
        hypergeometric,
        tl.tensor([[np.nextafter(tl.tensor(float(max(n, M))), tl.tensor(-1.0))]]),
    )
    tc.assertRaises(ValueError, log_likelihood, hypergeometric, tl.tensor([[5.5]]))

def test_update_backend(do_for_all_backends):
    torch.set_default_dtype(torch.float64)
    backends = ["numpy", "pytorch"]
    N = 15
    M = 10
    n = 10

    hypergeometric = Hypergeometric(Scope([0]), N, M, n)

    # create dummy input data (batch size x random variables)
    data = np.array([[5], [10]])

    log_probs = log_likelihood(hypergeometric, tl.tensor(data))

    # make sure that probabilities match python backend probabilities
    for backend in backends:
        with tl.backend_context(backend):
            hypergeometric_updated = updateBackend(hypergeometric)
            log_probs_updated = log_likelihood(hypergeometric_updated, tl.tensor(data))
            # check conversion from torch to python
            tc.assertTrue(np.allclose(tl_toNumpy(log_probs), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
