import random
import unittest

import torch
import numpy as np
import tensorly as tl

from spflow.meta.data import Scope
from spflow.torch.inference import log_likelihood
from spflow.tensorly.structure.general.layers.leaves.parametric.general_gaussian import GaussianLayer
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_gaussian import Gaussian
from spflow.torch.structure.general.layers.leaves.parametric.gaussian import updateBackend
from spflow.tensorly.utils.helper_functions import tl_toNumpy


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_layer_likelihood(self):

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

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)
        nodes_ll = torch.concat([log_likelihood(node, dummy_data) for node in nodes], dim=1)

        self.assertTrue(torch.allclose(layer_ll, nodes_ll))

    def test_gradient_computation(self):

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

        self.assertTrue(torch_gaussian.mean.grad is not None)
        self.assertTrue(torch_gaussian.std_aux.grad is not None)

        mean_orig = torch_gaussian.mean.detach().clone()
        std_aux_orig = torch_gaussian.std_aux.detach().clone()

        optimizer = torch.optim.SGD(torch_gaussian.parameters(), lr=1)
        optimizer.step()

        # make sure that parameters are correctly updated
        self.assertTrue(torch.allclose(mean_orig - torch_gaussian.mean.grad, torch_gaussian.mean))
        self.assertTrue(
            torch.allclose(
                std_aux_orig - torch_gaussian.std_aux.grad,
                torch_gaussian.std_aux,
            )
        )

        # verify that distribution parameters match parameters
        self.assertTrue(torch.allclose(torch_gaussian.mean, torch_gaussian.dist().mean))
        self.assertTrue(torch.allclose(torch_gaussian.std, torch_gaussian.dist().stddev))

    def test_gradient_optimization(self):

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

        self.assertTrue(
            torch.allclose(
                torch_gaussian.mean,
                torch.tensor([0.0, 0.0]),
                atol=1e-3,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch_gaussian.std,
                torch.tensor([1.0, 1.0]),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_likelihood_marginalization(self):

        gaussian = GaussianLayer(
            scope=[Scope([0]), Scope([1])],
            mean=random.random(),
            std=random.random() + 1e-7,
        )
        data = torch.tensor([[float("nan"), float("nan")]])

        # should not raise and error and should return 1
        probs = log_likelihood(gaussian, data).exp()

        self.assertTrue(torch.allclose(probs, torch.tensor([1.0, 1.0])))

    def test_support(self):
        # TODO
        pass

    def test_update_backend(self):
        backends = ["numpy", "pytorch"]
        layer = GaussianLayer(
            scope=[Scope([0]), Scope([1]), Scope([0])],
            mean=[0.2, 1.0, 2.3],
            std=[1.0, 0.3, 0.97],
        )

        dummy_data = torch.tensor([[0.5, 1.3], [3.9, 0.71], [1.0, 1.0]])

        layer_ll = log_likelihood(layer, dummy_data)

        # make sure that probabilities match python backend probabilities
        for backend in backends:
            tl.set_backend(backend)
            layer_updated = updateBackend(layer)
            log_probs_updated = log_likelihood(layer_updated, tl.tensor(dummy_data))
            # check conversion from torch to python
            self.assertTrue(np.allclose(tl_toNumpy(layer_ll), tl_toNumpy(log_probs_updated)))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
