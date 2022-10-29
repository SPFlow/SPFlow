from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.node import marginalize
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
)
from typing import Callable

import numpy as np
import unittest
import random


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        log_normal = CondLogNormal(Scope([0]))
        self.assertTrue(log_normal.cond_f is None)
        log_normal = CondLogNormal(
            Scope([0]), cond_f=lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(log_normal.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondLogNormal, Scope([]))
        self.assertRaises(Exception, CondLogNormal, Scope([0, 1]))
        self.assertRaises(Exception, CondLogNormal, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), stdev in (0,inf)

        log_normal = CondLogNormal(Scope([0]))

        # mean = inf and mean = nan
        log_normal.set_cond_f(lambda data: {"mean": np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": -np.inf, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": np.nan, "std": 1.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = 0 and stdev < 0
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": 0.0})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {"mean": 0.0, "std": np.nextafter(0.0, -1.0)}
        )
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # stdev = inf and stdev = nan
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": np.inf})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": -np.inf})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(lambda data: {"mean": 0.0, "std": np.nan})
        self.assertRaises(
            ValueError,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # invalid scopes
        self.assertRaises(Exception, CondLogNormal, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, CondLogNormal, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, CondLogNormal, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        log_normal = CondLogNormal(Scope([0]))

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)


if __name__ == "__main__":
    unittest.main()
