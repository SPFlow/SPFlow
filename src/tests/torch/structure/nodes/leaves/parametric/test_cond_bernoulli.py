from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_bernoulli import (
    CondBernoulli as BaseCondBernoulli,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_bernoulli import (
    CondBernoulli,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestBernoulli(unittest.TestCase):
    def test_initialization(self):

        bernoulli = CondBernoulli(Scope([0]))
        self.assertTrue(bernoulli.cond_f is None)
        bernoulli = CondBernoulli(Scope([0]), lambda x: {"p": 0.5})
        self.assertTrue(isinstance(bernoulli.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondBernoulli, Scope([]))
        self.assertRaises(Exception, CondBernoulli, Scope([0, 1]))
        self.assertRaises(Exception, CondBernoulli, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Bernoulli distribution: p in [0,1]

        bernoulli = CondBernoulli(Scope([0]))

        # p = 0
        bernoulli.set_cond_f(lambda data: {"p": 0.0})
        self.assertTrue(
            bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 0.0
        )
        # p = 1
        bernoulli.set_cond_f(lambda data: {"p": 1.0})
        self.assertTrue(
            bernoulli.retrieve_params(np.array([[1.0]]), DispatchContext())
            == 1.0
        )
        # p < 0 and p > 1
        bernoulli.set_cond_f(
            lambda data: {
                "p": torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
            }
        )
        self.assertRaises(
            Exception,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        bernoulli.set_cond_f(
            lambda data: {
                "p": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0))
            }
        )
        self.assertRaises(
            Exception,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # p = inf and p = nan
        bernoulli.set_cond_f(lambda data: {"p": np.inf})
        self.assertRaises(
            Exception,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        bernoulli.set_cond_f(lambda data: {"p": np.nan})
        self.assertRaises(
            Exception,
            bernoulli.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_structural_marginalization(self):

        bernoulli = CondBernoulli(Scope([0]))

        self.assertTrue(marginalize(bernoulli, [1]) is not None)
        self.assertTrue(marginalize(bernoulli, [0]) is None)

    def test_base_backend_conversion(self):

        p = random.random()

        # test scope instead
        torch_bernoulli = CondBernoulli(Scope([0]), p)
        node_bernoulli = BaseCondBernoulli(Scope([0]), p)

        # check conversion from torch to python
        self.assertTrue(
            np.all(
                torch_bernoulli.scopes_out == toBase(torch_bernoulli).scopes_out
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(
                node_bernoulli.scopes_out == toTorch(node_bernoulli).scopes_out
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
