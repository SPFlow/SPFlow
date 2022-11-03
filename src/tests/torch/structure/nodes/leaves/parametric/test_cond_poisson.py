from spflow.meta.data import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson as BaseCondPoisson,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_poisson import (
    CondPoisson,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.nodes.sum_node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestPoisson(unittest.TestCase):
    def test_initialization(self):

        poisson = CondPoisson(Scope([0], [1]))
        self.assertTrue(poisson.cond_f is None)
        poisson = CondPoisson(Scope([0], [1]), cond_f=lambda x: {"l": 1.0})
        self.assertTrue(isinstance(poisson.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondPoisson, Scope([]))
        self.assertRaises(Exception, CondPoisson, Scope([0, 1], [2]))
        self.assertRaises(Exception, CondPoisson, Scope([0]))

    def test_retrieve_params(self):

        # Valid parameters for Poisson distribution: l in (0,inf) (note: 0 included in pytorch)

        poisson = CondPoisson(Scope([0], [1]))

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

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondPoisson.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # Poisson feature type class
        self.assertTrue(
            CondPoisson.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])]
            )
        )

        # Poisson feature type instance
        self.assertTrue(
            CondPoisson.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondPoisson.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondPoisson.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondPoisson.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Discrete, FeatureTypes.Discrete],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        poisson = CondPoisson.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
        )
        poisson = CondPoisson.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])]
        )
        poisson = CondPoisson.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(l=1.5)])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondPoisson.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondPoisson.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondPoisson.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondPoisson))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondPoisson,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        poisson = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Poisson(l=1.5)])]
        )
        self.assertTrue(isinstance(poisson, CondPoisson))

    def test_structural_marginalization(self):

        poisson = CondPoisson(Scope([0], [1]))

        self.assertTrue(marginalize(poisson, [1]) is not None)
        self.assertTrue(marginalize(poisson, [0]) is None)

    def test_base_backend_conversion(self):

        torch_poisson = CondPoisson(Scope([0], [1]))
        node_poisson = BaseCondPoisson(Scope([0], [1]))

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
