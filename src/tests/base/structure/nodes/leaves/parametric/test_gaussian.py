from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random
import math


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Gaussian distribution: mean in (-inf,inf), stdev > 0

        mean = random.random()

        # mean = inf and mean = nan
        self.assertRaises(Exception, Gaussian, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, Gaussian, Scope([0]), np.nan, 1.0)

        # stdev = 0 and stdev < 0
        self.assertRaises(Exception, Gaussian, Scope([0]), mean, 0.0)
        self.assertRaises(
            Exception, Gaussian, Scope([0]), mean, np.nextafter(0.0, -1.0)
        )
        # stdev = inf and stdev = nan
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


if __name__ == "__main__":
    unittest.main()
