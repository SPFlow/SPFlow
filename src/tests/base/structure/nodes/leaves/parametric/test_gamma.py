from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.gamma import Gamma
from spflow.base.inference.nodes.leaves.parametric.gamma import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        # alpha > 0
        Gamma(Scope([0]), np.nextafter(0.0, 1.0), 1.0)
        # alpha = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 0.0, 1.0)
        # alpha < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), np.nextafter(0.0, -1.0), 1.0
        )
        # alpha = inf and alpha = nan
        self.assertRaises(Exception, Gamma, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0]), np.nan, 1.0)

        # beta > 0
        Gamma(Scope([0]), 1.0, np.nextafter(0.0, 1.0))
        # beta = 0
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, 0.0)
        # beta < 0
        self.assertRaises(
            Exception, Gamma, Scope([0]), 1.0, np.nextafter(0.0, -1.0)
        )
        # beta = inf and beta = nan
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.inf)
        self.assertRaises(Exception, Gamma, Scope([0]), 1.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Gamma, Scope([]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0, 1]), 1.0, 1.0)
        self.assertRaises(Exception, Gamma, Scope([0], [1]), 1.0, 1.0)

    def test_structural_marginalization(self):

        gamma = Gamma(Scope([0]), 1.0, 1.0)

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)


if __name__ == "__main__":
    unittest.main()
