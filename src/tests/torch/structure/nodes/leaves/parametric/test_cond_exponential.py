from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential as BaseCondExponential,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_exponential import (
    CondExponential,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        exponential = CondExponential(Scope([0]))
        self.assertTrue(exponential.cond_f is None)
        exponential = CondExponential(Scope([0]), lambda x: {"l": 0.5})
        self.assertTrue(isinstance(exponential.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondExponential, Scope([]))
        self.assertRaises(Exception, CondExponential, Scope([0, 1]))
        self.assertRaises(Exception, CondExponential, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Exponential distribution: l>0

        exponential = CondExponential(Scope([0]))

        # l > 0
        exponential.set_cond_f(
            lambda data: {
                "l": torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
            }
        )
        self.assertTrue(
            exponential.retrieve_params(np.array([[1.0]]), DispatchContext())
            == torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )

        # l = 0 and l < 0
        exponential.set_cond_f(lambda data: {"l": torch.tensor(0.0)})
        self.assertRaises(
            Exception,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        exponential.set_cond_f(
            lambda data: {
                "l": torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))
            }
        )
        self.assertRaises(
            Exception,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # l = inf and l = nan
        exponential.set_cond_f(lambda data: {"l": torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        exponential.set_cond_f(lambda data: {"l": torch.tensor(float("nan"))})
        self.assertRaises(
            Exception,
            exponential.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # invalid scopes
        self.assertRaises(Exception, CondExponential, Scope([]))
        self.assertRaises(Exception, CondExponential, Scope([0, 1]))
        self.assertRaises(Exception, CondExponential, Scope([0], [1]))

    def test_structural_marginalization(self):

        exponential = CondExponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)

    def test_base_backend_conversion(self):

        l = random.random()

        torch_exponential = CondExponential(Scope([0]))
        node_exponential = BaseCondExponential(Scope([0]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(
                torch_exponential.scopes_out
                == toBase(torch_exponential).scopes_out
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(
                node_exponential.scopes_out
                == toTorch(node_exponential).scopes_out
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
