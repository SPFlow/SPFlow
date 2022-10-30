from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.poisson import Poisson
from spflow.base.inference.nodes.leaves.parametric.poisson import log_likelihood
from spflow.base.inference.module import likelihood

import numpy as np
import unittest
import random


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in scipy)

        # l = 0
        Poisson(Scope([0]), 0.0)
        # l > 0
        Poisson(Scope([0]), np.nextafter(0.0, 1.0))
        # l = -inf and l = inf
        self.assertRaises(Exception, Poisson, Scope([0]), -np.inf)
        self.assertRaises(Exception, Poisson, Scope([0]), np.inf)
        # l = nan
        self.assertRaises(Exception, Poisson, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Poisson, Scope([]), 1)
        self.assertRaises(Exception, Poisson, Scope([0, 1]), 1)
        self.assertRaises(Exception, Poisson, Scope([0], [1]), 1)

    def test_structural_marginalization(self):

        poisson = Poisson(Scope([0]), 1.0)

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)


if __name__ == "__main__":
    unittest.main()
