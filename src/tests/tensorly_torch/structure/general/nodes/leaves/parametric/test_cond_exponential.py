import random
import unittest
from typing import Callable

import numpy as np
import torch

from spflow.base.structure.general.nodes.leaves.parametric.cond_exponential import (
    CondExponential as BaseCondExponential,
)
from spflow.meta.data import Scope
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.tensorly.structure.autoleaf import AutoLeaf
from spflow.torch.structure.spn import CondExponential as TorchCondExponential
from spflow.tensorly.structure.general.nodes.leaves.parametric.general_cond_exponential import CondExponential
from spflow.torch.structure.general.nodes.leaves.parametric.cond_exponential import (
    #CondExponential,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.nodes.sum_node import marginalize


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        exponential = CondExponential(Scope([0], [1]))
        self.assertTrue(exponential.cond_f is None)
        exponential = CondExponential(Scope([0], [1]), lambda x: {"l": 0.5})
        self.assertTrue(isinstance(exponential.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondExponential, Scope([]))
        self.assertRaises(Exception, CondExponential, Scope([0, 1], [2]))
        self.assertRaises(Exception, CondExponential, Scope([0]))

    def test_retrieve_params(self):

        # Valid parameters for Exponential distribution: l>0

        exponential = CondExponential(Scope([0], [1]))

        # l > 0
        exponential.set_cond_f(lambda data: {"l": torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))})
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

        exponential.set_cond_f(lambda data: {"l": torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0))})
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

    def test_accept(self):

        # continuous meta type
        self.assertTrue(CondExponential.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]))

        # Exponential feature type class
        self.assertTrue(CondExponential.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential])]))

        # Exponential feature type instance
        self.assertTrue(CondExponential.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential(1.0)])]))

        # invalid feature type
        self.assertFalse(CondExponential.accepts([FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]))

        # non-conditional scope
        self.assertFalse(CondExponential.accepts([FeatureContext(Scope([0]), [FeatureTypes.Continuous])]))

        # multivariate signature
        self.assertFalse(
            CondExponential.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondExponential.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])])
        CondExponential.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential])])
        CondExponential.from_signatures([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential(l=1.5)])])

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondExponential.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondExponential.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondExponential.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondExponential))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondExponential,
            AutoLeaf.infer([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential])]),
        )

        # make sure AutoLeaf can return correctly instantiated object
        exponential = AutoLeaf([FeatureContext(Scope([0], [1]), [FeatureTypes.Exponential])])
        self.assertTrue(isinstance(exponential, TorchCondExponential))

    def test_structural_marginalization(self):

        exponential = CondExponential(Scope([0], [1]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)

    def test_base_backend_conversion(self):

        l = random.random()

        torch_exponential = CondExponential(Scope([0], [1]))
        node_exponential = BaseCondExponential(Scope([0], [1]))

        # check conversion from torch to python
        self.assertTrue(np.all(torch_exponential.scopes_out == toBase(torch_exponential).scopes_out))
        # check conversion from python to torch
        self.assertTrue(np.all(node_exponential.scopes_out == toTorch(node_exponential).scopes_out))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
