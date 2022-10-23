from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma as BaseGamma
from spflow.base.inference.nodes.leaves.parametric.gamma import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.gamma import Gamma, toBase, toTorch
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.gamma import log_likelihood
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest

class TestGamma(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        # alpha > 0
        Gamma(Scope([0]), torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)), 1.0)
        # alpha = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 0.0, 1.0)
        # alpha < 0
        self.assertRaises(Exception, Gamma, Scope([0]), np.nextafter(0.0, -1.0), 1.0)
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, Gamma, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0]), np.nan, 1.0)

        # beta > 0
        Gamma(Scope([0]), 1.0, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
        # beta = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, 0.0)
        # beta < 0
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nextafter(0.0, -1.0))
        # beta = inf and beta = non
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.inf)
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gamma, Scope([]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0, 1]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0],[1]), 1.0, 1.0)
    
    def test_structural_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)
    
    def test_base_backend_conversion(self):

        alpha = random.randint(1, 5)
        beta = random.randint(1, 5)

        torch_gamma = Gamma(Scope([0]), alpha, beta)
        node_gamma = BaseGamma(Scope([0]), alpha, beta)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_gamma.get_params()]),
                np.array([*toBase(torch_gamma).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_gamma.get_params()]), np.array([*toTorch(node_gamma).get_params()])
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
