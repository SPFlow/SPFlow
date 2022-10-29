from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_gaussian import (
    CondGaussian,
)
from typing import Callable

import numpy as np
import unittest
import random
import math


class TestGaussian(unittest.TestCase):
    def test_initialization(self):

        gaussian = CondGaussian(Scope([0]))
        self.assertTrue(gaussian.cond_f is None)
        gaussian = CondGaussian(
            Scope([0]), cond_f=lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(gaussian.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGaussian, Scope([]))
        self.assertRaises(Exception, CondGaussian, Scope([0, 1]))
        self.assertRaises(Exception, CondGaussian, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Gaussian distribution: mean in (-inf,inf), stdev > 0
        gaussian = CondGaussian(Scope([0]))

        # mean = inf and mean = nan
        gaussian.set_cond_f(lambda data: {"mean": np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": -np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": np.nan, "std": 1.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = 0 and stdev < 0
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": 0.0})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(
            lambda data: {"mean": 0.0, "std": np.nextafter(0.0, -1.0)}
        )
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = inf and stdev = nan
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": np.inf})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": -np.inf})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gaussian.set_cond_f(lambda data: {"mean": 0.0, "std": np.nan})
        self.assertRaises(
            ValueError,
            gaussian.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # invalid scopes
        self.assertRaises(Exception, CondGaussian, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, CondGaussian, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, CondGaussian, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        gaussian = CondGaussian(Scope([0]))

        self.assertTrue(marginalize(gaussian, [1]) is not None)
        self.assertTrue(marginalize(gaussian, [0]) is None)


if __name__ == "__main__":
    unittest.main()
