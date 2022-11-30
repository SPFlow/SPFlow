import unittest

import numpy as np
import torch

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.torch.structure.spn import MultivariateGaussian, MultivariateGaussianLayer


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

        mean_values = [
            torch.zeros(2),
            torch.arange(3, dtype=torch.get_default_dtype()),
        ]
        cov_values = [
            torch.eye(2),
            torch.tensor(
                [
                    [2, 2, 1],
                    [2, 3, 2],
                    [1, 2, 3],
                ],
                dtype=torch.get_default_dtype(),
            ),
        ]

        layer = MultivariateGaussianLayer(
            scope=[Scope([0, 1]), Scope([2, 3, 4])],
            mean=mean_values,
            cov=cov_values,
        )

        nodes = [
            MultivariateGaussian(
                Scope([0, 1]), mean=mean_values[0], cov=cov_values[0]
            ),
            MultivariateGaussian(
                Scope([2, 3, 4]), mean=mean_values[1], cov=cov_values[1]
            ),
        ]

        dummy_data = torch.vstack([torch.zeros(5), torch.ones(5)])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat(
            [log_likelihood(node, dummy_data) for node in nodes], dim=1
        )

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

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

        self.assertTrue(
            all(
                [
                    n.mean.grad is not None
                    for n in torch_multivariate_gaussian.nodes
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    n.tril_diag_aux.grad is not None
                    for n in torch_multivariate_gaussian.nodes
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    n.tril_nondiag.grad is not None
                    for n in torch_multivariate_gaussian.nodes
                ]
            )
        )

        mean_orig = [
            node.mean.detach().clone()
            for node in torch_multivariate_gaussian.nodes
        ]
        tril_diag_aux_orig = [
            node.tril_diag_aux.detach().clone()
            for node in torch_multivariate_gaussian.nodes
        ]
        tril_nondiag_orig = [
            node.tril_nondiag.detach().clone()
            for node in torch_multivariate_gaussian.nodes
        ]

        optimizer = torch.optim.SGD(
            torch_multivariate_gaussian.parameters(), lr=1
        )
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(
            all(
                [
                    torch.allclose(m_orig - m_current.grad, m_current)
                    for m_orig, m_current in zip(
                        mean_orig, torch_multivariate_gaussian.mean
                    )
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    torch.allclose(t_orig - t_current.grad, t_current)
                    for t_orig, t_current in zip(
                        tril_diag_aux_orig,
                        [
                            n.tril_diag_aux
                            for n in torch_multivariate_gaussian.nodes
                        ],
                    )
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    torch.allclose(t_orig - t_current.grad, t_current)
                    for t_orig, t_current in zip(
                        tril_nondiag_orig,
                        [
                            n.tril_nondiag
                            for n in torch_multivariate_gaussian.nodes
                        ],
                    )
                ]
            )
        )

    def test_gradient_optimization(self):
        # can be ommited here: as long as the gradient computation is correct, we only need the nodes can be optimized correctly which is tested separately
        pass

    def test_likelihood_marginalization(self):

        gaussian = MultivariateGaussianLayer(
            scope=[Scope([0, 1]), Scope([1, 2])],
            mean=torch.zeros(2, 2),
            cov=torch.stack([torch.eye(2), torch.eye(2)]),
        )
        data = torch.tensor([[float("nan"), float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(gaussian, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
