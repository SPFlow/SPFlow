from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal as BaseCondLogNormal,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_log_normal import (
    CondLogNormal,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        log_normal = CondLogNormal(Scope([0]))
        self.assertTrue(log_normal.cond_f is None)
        log_normal = CondLogNormal(
            Scope([0]), lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(log_normal.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondLogNormal, Scope([]))
        self.assertRaises(Exception, CondLogNormal, Scope([0, 1]))
        self.assertRaises(Exception, CondLogNormal, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), std in (0,inf)

        log_normal = CondLogNormal(Scope([0]))

        # mean = +-inf and mean = 0
        log_normal.set_cond_f(
            lambda data: {
                "mean": torch.tensor(float("inf")),
                "std": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {
                "mean": -torch.tensor(float("inf")),
                "std": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {
                "mean": torch.tensor(float("nan")),
                "std": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # std <= 0
        log_normal.set_cond_f(
            lambda data: {"mean": torch.tensor(0.0), "std": torch.tensor(0.0)}
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {
                "mean": torch.tensor(0.0),
                "std": torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # std = +-inf and std = nan
        log_normal.set_cond_f(
            lambda data: {
                "mean": torch.tensor(0.0),
                "std": -torch.tensor(float("inf")),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        log_normal.set_cond_f(
            lambda data: {
                "mean": torch.tensor(0.0),
                "std": torch.tensor(float("nan")),
            }
        )
        self.assertRaises(
            Exception,
            log_normal.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_structural_marginalization(self):

        log_normal = CondLogNormal(Scope([0]))

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)

    def test_base_backend_conversion(self):

        torch_log_normal = CondLogNormal(Scope([0]))
        node_log_normal = BaseCondLogNormal(Scope([0]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(
                torch_log_normal.scopes_out
                == toBase(torch_log_normal).scopes_out
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(
                node_log_normal.scopes_out
                == toTorch(node_log_normal).scopes_out
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
