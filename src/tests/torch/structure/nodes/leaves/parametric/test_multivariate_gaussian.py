from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian as BaseMultivariateGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.node import SPNProductNode
from spflow.torch.inference.nodes.node import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
    toBase,
    toTorch,
    marginalize,
)
from spflow.torch.inference.nodes.leaves.parametric.multivariate_gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import math

import unittest


class TestMultivariateGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Multivariate Gaussian distribution: mean vector in R^k, covariance matrix in R^(k x k) symmetric positive semi-definite

        # mean contains inf and mean contains nan
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.tensor([0.0, float("inf")]),
            torch.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.tensor([-float("inf"), 0.0]),
            torch.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.tensor([0.0, float("nan")]),
            torch.eye(2),
        )

        # mean vector of wrong shape
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(3),
            torch.eye(2),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros((1, 1, 2)),
            torch.eye(2),
        )

        # covariance matrix of wrong shape
        M = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertRaises(
            Exception, MultivariateGaussian, Scope([0, 1]), torch.zeros(2), M
        )
        self.assertRaises(
            Exception, MultivariateGaussian, Scope([0, 1]), torch.zeros(2), M.T
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(2),
            np.eye(3),
        )
        # covariance matrix not symmetric positive semi-definite
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(2),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(2),
            -torch.eye(2),
        )
        # covariance matrix containing inf or nan
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(2),
            torch.tensor([[float("inf"), 0], [0, float("inf")]]),
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1]),
            torch.zeros(2),
            torch.tensor([[float("nan"), 0], [0, float("nan")]]),
        )

        # duplicate scope variables
        self.assertRaises(
            Exception, Scope, [0, 0]
        )  # makes sure that MultivariateGaussian can also not be given a scope with duplicate query variables

        # invalid scopes
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1, 2]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertRaises(
            Exception,
            MultivariateGaussian,
            Scope([0, 1], [2]),
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        # initialize using lists
        MultivariateGaussian(
            Scope([0, 1]), [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
        )

        # initialize using numpy arrays
        MultivariateGaussian(Scope([0, 1]), np.zeros(2), np.eye(2))

    def test_structural_marginalization(self):

        multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1]), [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
        )

        self.assertTrue(
            isinstance(
                marginalize(multivariate_gaussian, [2]), MultivariateGaussian
            )
        )
        self.assertTrue(
            isinstance(marginalize(multivariate_gaussian, [1]), Gaussian)
        )
        self.assertTrue(marginalize(multivariate_gaussian, [0, 1]) is None)

    def test_base_backend_conversion(self):

        mean = np.arange(3)
        cov = np.array([[2, 2, 1], [2, 3, 2], [1, 2, 3]])

        torch_multivariate_gaussian = MultivariateGaussian(
            Scope([0, 1, 2]), mean, cov
        )
        node_multivariate_gaussian = BaseMultivariateGaussian(
            Scope([0, 1, 2]), mean.tolist(), cov.tolist()
        )

        node_params = node_multivariate_gaussian.get_params()
        torch_params = torch_multivariate_gaussian.get_params()

        # check conversion from torch to python
        torch_to_node_params = toBase(torch_multivariate_gaussian).get_params()

        self.assertTrue(
            np.allclose(
                np.array([torch_params[0]]),
                np.array([torch_to_node_params[0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.array([torch_params[1]]),
                np.array([torch_to_node_params[1]]),
            )
        )
        # check conversion from python to torch#
        node_to_torch_params = toTorch(node_multivariate_gaussian).get_params()

        self.assertTrue(
            np.allclose(
                np.array([node_params[0]]),
                np.array([node_to_torch_params[0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                np.array([node_params[1]]),
                np.array([node_to_torch_params[1]]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
