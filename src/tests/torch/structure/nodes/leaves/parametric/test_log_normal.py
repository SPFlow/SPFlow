from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.log_normal import (
    LogNormal as BaseLogNormal,
)
from spflow.base.inference.nodes.leaves.parametric.log_normal import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.log_normal import (
    LogNormal,
    toBase,
    toTorch,
)
from spflow.torch.structure.spn.nodes.sum_node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.log_normal import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestLogNormal(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Log-Normal distribution: mean in (-inf,inf), std in (0,inf)

        # mean = +-inf and mean = 0
        self.assertRaises(Exception, LogNormal, Scope([0]), np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), -np.inf, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0]), np.nan, 1.0)

        mean = random.random()

        # std <= 0
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, 0.0)
        self.assertRaises(
            Exception, LogNormal, Scope([0]), mean, np.nextafter(0.0, -1.0)
        )
        # std = +-inf and std = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, -np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, LogNormal, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)

    def test_accept(self):

        # continuous meta type
        self.assertTrue(
            LogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
            )
        )

        # LogNormal feature type class
        self.assertTrue(
            LogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.LogNormal])]
            )
        )

        # LogNormal feature type instance
        self.assertTrue(
            LogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.LogNormal(0.0, 1.0)])]
            )
        )

        # invalid feature type
        self.assertFalse(
            LogNormal.accepts(
                [FeatureContext(Scope([0]), [FeatureTypes.Discrete])]
            )
        )

        # conditional scope
        self.assertFalse(
            LogNormal.accepts(
                [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])]
            )
        )

        # multivariate signature
        self.assertFalse(
            LogNormal.accepts(
                [
                    FeatureContext(
                        Scope([0, 1]),
                        [FeatureTypes.Continuous, FeatureTypes.Continuous],
                    )
                ]
            )
        )

    def test_initialization_from_signatures(self):

        log_normal = LogNormal.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.Continuous])]
        )
        self.assertTrue(torch.isclose(log_normal.mean, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(log_normal.std, torch.tensor(1.0)))

        log_normal = LogNormal.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.LogNormal])]
        )
        self.assertTrue(torch.isclose(log_normal.mean, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(log_normal.std, torch.tensor(1.0)))

        log_normal = LogNormal.from_signatures(
            [FeatureContext(Scope([0]), [FeatureTypes.LogNormal(-1.0, 1.5)])]
        )
        self.assertTrue(torch.isclose(log_normal.mean, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(log_normal.std, torch.tensor(1.5)))

        # ----- invalid arguments -----

        # invalid feature type
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [FeatureContext(Scope([0]), [FeatureTypes.Discrete])],
        )

        # conditional scope
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [FeatureContext(Scope([0], [1]), [FeatureTypes.Continuous])],
        )

        # multivariate signature
        self.assertRaises(
            ValueError,
            LogNormal.from_signatures,
            [
                FeatureContext(
                    Scope([0, 1]),
                    [FeatureTypes.Continuous, FeatureTypes.Continuous],
                )
            ],
        )

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(LogNormal))

        # make sure leaf is correctly inferred
        self.assertEqual(
            LogNormal,
            AutoLeaf.infer(
                [FeatureContext(Scope([0]), [FeatureTypes.LogNormal])]
            ),
        )

        # make sure AutoLeaf can return correctly instantiated object
        log_normal = AutoLeaf(
            [
                FeatureContext(
                    Scope([0]), [FeatureTypes.LogNormal(mean=-1.0, std=0.5)]
                )
            ]
        )
        self.assertTrue(isinstance(log_normal, LogNormal))
        self.assertTrue(torch.isclose(log_normal.mean, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(log_normal.std, torch.tensor(0.5)))

    def test_base_backend_conversion(self):

        mean = random.random()
        std = random.random() + 1e-7  # offset by small number to avoid zero

        torch_log_normal = LogNormal(Scope([0]), mean, std)
        node_log_normal = BaseLogNormal(Scope([0]), mean, std)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_log_normal.get_params()]),
                np.array([*toBase(torch_log_normal).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_log_normal.get_params()]),
                np.array([*toTorch(node_log_normal).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
