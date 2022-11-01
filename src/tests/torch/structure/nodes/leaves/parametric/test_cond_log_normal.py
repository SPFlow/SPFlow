from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.meta.dispatch.dispatch_context import DispatchContext
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

        log_normal = CondLogNormal(Scope([0], [1]))
        self.assertTrue(log_normal.cond_f is None)
        log_normal = CondLogNormal(
            Scope([0], [1]), lambda x: {"mean": 0.0, "std": 1.0}
        )
        self.assertTrue(isinstance(log_normal.cond_f, Callable))

        # invalid scopes
        self.assertRaises(Exception, CondLogNormal, Scope([]))
        self.assertRaises(Exception, CondLogNormal, Scope([0, 1], [2]))
        self.assertRaises(Exception, CondLogNormal, Scope([0]))

    def test_retrieve_params(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), std in (0,inf)

        log_normal = CondLogNormal(Scope([0], [1]))

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

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # LogNormal feature type class
        self.assertTrue(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
            )
        )

        # LogNormal feature type instance
        self.assertTrue(
            CondLogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0], [1]), [FeatureTypes.LogNormal(0.0, 1.0)]
                    )
                ]
            )
        )

        # invalid feature type
        self.assertFalse(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])]
            )
        )

        # non-conditional scope
        self.assertFalse(
            CondLogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            CondLogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0, 1], [2]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        CondLogNormal.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
        )
        CondLogNormal.from_signatures(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
        )
        CondLogNormal.from_signatures(
            [
                FeatureContext(
                    Scope([0], [1]), [FeatureTypes.LogNormal(-1.0, 1.5)]
                )
            ]
        )

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Discrete])],
        )

        # non-conditional scope
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            CondLogNormal.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1], [2]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(CondLogNormal))

        # make sure leaf is correctly inferred
        self.assertEqual(
            CondLogNormal,
            AutoLeaf.infer(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf(
            [FeatureContext(Scope([0], [1]), [FeatureTypes.LogNormal])]
        )
        self.assertTrue(isinstance(log_normal, CondLogNormal))

    def test_structural_marginalization(self):

        log_normal = CondLogNormal(Scope([0], [1]))

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)

    def test_base_backend_conversion(self):

        torch_log_normal = CondLogNormal(Scope([0], [1]))
        node_log_normal = BaseCondLogNormal(Scope([0], [1]))

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
