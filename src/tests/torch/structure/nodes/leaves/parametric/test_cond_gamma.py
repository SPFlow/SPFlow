from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.meta.dispatch.dispatch_context import DispatchContext
from spflow.base.structure.nodes.leaves.parametric.cond_gamma import (
    CondGamma as BaseCondGamma,
)
from spflow.torch.structure.nodes.leaves.parametric.cond_gamma import (
    CondGamma,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from typing import Callable

import torch
import numpy as np

import random
import unittest


class TestGamma(unittest.TestCase):
    def test_initialization(self):

        gamma = CondGamma(Scope([0], [1]))
        self.assertTrue(gamma.cond_f is None)
        gamma = CondGamma(
            Scope([0], [1]), lambda x: {"alpha": 1.0, "beta": 1.0}
        )
        self.assertTrue(isinstance(gamma.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondGamma, Scope([]))
        self.assertRaises(Exception, CondGamma, Scope([0, 1], [2]))
        self.assertRaises(Exception, CondGamma, Scope([0]))

    def test_retrieve_params(self):

        # Valid parameters for Gamma distribution: alpha>0, beta>0

        gamma = CondGamma(Scope([0], [1]))

        # alpha > 0
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)),
                "beta": torch.tensor(1.0),
            }
        )
        self.assertTrue(
            gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[0]
            == torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # alpha = 0
        gamma.set_cond_f(
            lambda data: {"alpha": torch.tensor(0.0), "beta": torch.tensor(1.0)}
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # alpha < 0
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0)),
                "beta": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # alpha = inf and alpha = nan
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(float("inf")),
                "beta": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(float("nan")),
                "beta": torch.tensor(1.0),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

        # beta > 0
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(1.0),
                "beta": torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)),
            }
        )
        self.assertTrue(
            gamma.retrieve_params(np.array([[1.0]]), DispatchContext())[1]
            == torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # beta = 0
        gamma.set_cond_f(
            lambda data: {"alpha": torch.tensor(1.0), "beta": torch.tensor(0.0)}
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # beta < 0
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(0.0),
                "beta": torch.nextafter(torch.tensor(0.0), -torch.tensor(1.0)),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        # beta = inf and beta = non
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(1.0),
                "beta": torch.tensor(float("inf")),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )
        gamma.set_cond_f(
            lambda data: {
                "alpha": torch.tensor(1.0),
                "beta": torch.tensor(float("nan")),
            }
        )
        self.assertRaises(
            Exception,
            gamma.retrieve_params,
            np.array([[1.0]]),
            DispatchContext(),
        )

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # Gamma feature type class
        self.assertTrue(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
            )
        )

        # Gamma feature type instance
        self.assertTrue(
            CondGamma.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.Gamma(1.0, 1.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondGamma.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondGamma.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondGamma.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
        )
        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
        )
        CondGamma.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma(1.5, 0.5)])]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondGamma.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondGamma))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondGamma,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        gamma = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Gamma])]
        )
        self.assertTrue(isinstance(gamma, CondGamma))

    def test_structural_marginalization(self):

        gamma = CondGamma(Scope([0], [1]))

        self.assertTrue(marginalize(gamma, [1]) is not None)
        self.assertTrue(marginalize(gamma, [0]) is None)

    def test_base_backend_conversion(self):

        torch_gamma = CondGamma(Scope([0], [1]))
        node_gamma = BaseCondGamma(Scope([0], [1]))

        # check conversion from torch to python
        self.assertTrue(
            np.all(torch_gamma.scopes_out == toBase(torch_gamma).scopes_out)
        )
        # check conversion from python to torch
        self.assertTrue(
            np.all(node_gamma.scopes_out == toTorch(node_gamma).scopes_out)
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
