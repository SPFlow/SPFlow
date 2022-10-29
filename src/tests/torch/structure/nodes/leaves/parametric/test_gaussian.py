from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.gaussian import (
    Gaussian as BaseGaussian,
)
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.gaussian import (
    Gaussian,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: mean in R, std > 0

        mean = random.random()

        # mean = inf and mean = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), np.nan, 1.0)

        # std = 0 and std < 0
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, 0.0)
        self.assertRaises(
            Exception, Gaussian, Scope([0]), mean, np.nextafter(0.0, -1.0)
        )
        # std = inf and std = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gaussian, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        gaussian = Gaussian(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)

    def test_base_backend_conversion(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_gaussian = Gaussian(Scope([0]), mean, std)
        node_gaussian = BaseGaussian(Scope([0]), mean, std)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gaussian.get_params()]),
                np.array([*toBase(torch_gaussian).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gaussian.get_params()]),
                np.array([*toTorch(node_gaussian).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
