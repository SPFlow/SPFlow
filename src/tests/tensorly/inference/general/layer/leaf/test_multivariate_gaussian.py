import unittest

import numpy as np
import torch
import tensorly as tl

from spflow.meta.data import Scope
from spflow.tensorly.inference import log_likelihood
from spflow.tensorly.structure.general.layer.leaf.general_multivariate_gaussian import MultivariateGaussianLayer
from spflow.tensorly.structure.general.node.leaf.general_multivariate_gaussian import MultivariateGaussian
from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy

tc = unittest.TestCase()

def test_layer_likelihood(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    mean_values = [
        tl.zeros(2),
        tl.arange(3, dtype=tl.float64),
    ]
    cov_values = [
        tl.eye(2),
        tl.tensor(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ],
            dtype=tl.float64,
        ),
    ]

    layer = MultivariateGaussianLayer(
        scope=[Scope([0, 1]), Scope([2, 3, 4])],
        mean=mean_values,
        cov=cov_values,
    )

    nodes = [
        MultivariateGaussian(Scope([0, 1]), mean=mean_values[0], cov=cov_values[0]),
        MultivariateGaussian(Scope([2, 3, 4]), mean=mean_values[1], cov=cov_values[1]),
    ]

    dummy_data = tl.tensor(np.vstack([tl.zeros(5), tl.ones(5)]))

    layer_ll = log_likelihood(layer, dummy_data)
    nodes_ll = tl.concatenate([log_likelihood(node, dummy_data) for node in nodes], axis=1)

    tc.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(nodes_ll)))

def test_gradient_computation(do_for_all_backends):
    torch.set_default_dtype(torch.float32)

    if do_for_all_backends == "numpy":
        return

    mean = [np.zeros(2), np.arange(3)]
    cov = [np.eye(2), np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])]

    torch_multivariate_gaussian = MultivariateGaussianLayer(
        scope=[Scope([0, 1]), Scope([2, 3, 4])], mean=mean, cov=cov
    )

    # create dummy input data (batch size x random variables)
    data = torch.randn(3, 5)

    log_probs_torch = log_likelihood(torch_multivariate_gaussian, data)

    # create dummy targets
    targets_torch = torch.ones(3, 2)

    loss = torch.nn.MSELoss()(log_probs_torch, targets_torch)
    loss.backward()

    tc.assertTrue(all([n.mean.grad is not None for n in torch_multivariate_gaussian.nodes]))
    tc.assertTrue(all([n.tril_diag_aux.grad is not None for n in torch_multivariate_gaussian.nodes]))
    tc.assertTrue(all([n.tril_nondiag.grad is not None for n in torch_multivariate_gaussian.nodes]))

    mean_orig = [node.mean.detach().clone() for node in torch_multivariate_gaussian.nodes]
    tril_diag_aux_orig = [node.tril_diag_aux.detach().clone() for node in torch_multivariate_gaussian.nodes]
    tril_nondiag_orig = [node.tril_nondiag.detach().clone() for node in torch_multivariate_gaussian.nodes]

    optimizer = torch.optim.SGD(torch_multivariate_gaussian.parameters(), lr=1)
    optimizer.step()

    # make sure that parameters are correctly updated
    tc.assertTrue(
        all(
            [
                torch.allclose(m_orig - m_current.grad, m_current)
                for m_orig, m_current in zip(mean_orig, torch_multivariate_gaussian.mean)
            ]
        )
    )
    tc.assertTrue(
        all(
            [
                torch.allclose(t_orig - t_current.grad, t_current)
                for t_orig, t_current in zip(
                    tril_diag_aux_orig,
                    [n.tril_diag_aux for n in torch_multivariate_gaussian.nodes],
                )
            ]
        )
    )
    tc.assertTrue(
        all(
            [
                torch.allclose(t_orig - t_current.grad, t_current)
                for t_orig, t_current in zip(
                    tril_nondiag_orig,
                    [n.tril_nondiag for n in torch_multivariate_gaussian.nodes],
                )
            ]
        )
    )

def test_gradient_optimization(do_for_all_backends):
    # can be ommited here: as long as the gradient computation is correct, we only need the nodes can be optimized correctly which is tested separately
    pass

def test_likelihood_marginalization(do_for_all_backends):

    gaussian = MultivariateGaussianLayer(
        scope=[Scope([0, 1]), Scope([1, 2])],
        mean=tl.zeros((2, 2)),
        cov=tl.stack([tl.eye(2), tl.eye(2)]),
    )
    data = tl.tensor([[float("nan"), float("nan"), float("nan")]])

    # should not raise and error and should return 1
    probs = tl.exp(log_likelihood(gaussian, data))

    tc.assertTrue(np.allclose(tl_toNumpy(probs), tl.tensor([1.0, 1.0])))

def test_support(do_for_all_backends):
    # TODO
    pass

def test_update_backend(do_for_all_backends):
    backends = ["numpy", "pytorch"]
    mean_values = [
        tl.zeros(2),
        tl.arange(3, dtype=tl.float64),
    ]
    cov_values = [
        tl.eye(2),
        tl.tensor(
            [
                [2, 2, 1],
                [2, 3, 2],
                [1, 2, 3],
            ],
            dtype=tl.float64,
        ),
    ]

    layer = MultivariateGaussianLayer(
        scope=[Scope([0, 1]), Scope([2, 3, 4])],
        mean=mean_values,
        cov=cov_values,
    )

    dummy_data = tl.tensor(np.vstack([tl.zeros(5), tl.ones(5)]))

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
