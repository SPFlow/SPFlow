from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson as BaseCondPoisson,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        poisson = CondPoisson(Scope([0]))
        self.assertTrue(poisson.cond_f is None)
        poisson = CondPoisson(Scope([0]), cond_f=lambda x: {"l": 1.0})
        self.assertTrue(isinstance(poisson.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondPoisson, Scope([]))
        self.assertRaises(Exception, CondPoisson, Scope([0, 1]))
        self.assertRaises(Exception, CondPoisson, Scope([0], [1]))

    def test_retrieve_params(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)

        poisson = CondPoisson(Scope([0]))

        # l = 0
        poisson.set_cond_f(lambda data: {"l": torch.tensor(0.0)})
        self.assertTrue(
            poisson.retrieve_params(np.array([[1.0]]), DispatchContext())
            == torch.tensor(0.0)
        )
        # l > 0
        poisson.set_cond_f(
            lambda data: {
                "l": torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
            }
        )
        self.assertTrue(
            poisson.retrieve_params(np.array([[1.0]]), DispatchContext())
            == torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # l = -inf and l = inf
        poisson.set_cond_f(lambda data: {"l": torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        poisson.set_cond_f(lambda data: {"l": -torch.tensor(float("inf"))})
        self.assertRaises(
            Exception,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # l = nan
        poisson.set_cond_f(lambda data: {"l": torch.tensor(float("nan"))})
        self.assertRaises(
            Exception,
            poisson.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_structural_marginalization(self):

        poisson = CondPoisson(Scope([0]))

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)

    def test_base_backend_conversion(self):

        torch_poisson = CondPoisson(Scope([0]))
        node_poisson = BaseCondPoisson(Scope([0]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(torch_poisson.scopes_out == toBase(torch_poisson).scopes_out)
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(node_poisson.scopes_out == toTorch(node_poisson).scopes_out)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
