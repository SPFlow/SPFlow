from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.torch.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.leaves.parametric.uniform import (
    Uniform as BaseUniform,
)
from spflow.base.inference.nodes.leaves.parametric.uniform import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.uniform import (
    Uniform,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.uniform import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestUniform(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Uniform distribution: a<b

        # start = end
        start_end = random.random()
        self.assertRaises(Exception, Uniform, Scope([0]), start_end, start_end)
        # start > end
        self.assertRaises(
            Exception,
            Uniform,
            Scope([0]),
            start_end,
            torch.nextafter(torch.tensor(start_end), torch.tensor(-1.0)),
        )
        # start = +-inf and start = nan
        self.assertRaises(Exception, Uniform, Scope([0]), np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), -np.inf, 0.0)
        self.assertRaises(Exception, Uniform, Scope([0]), np.nan, 0.0)
        # end = +-inf and end = nan
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, -np.inf)
        self.assertRaises(Exception, Uniform, Scope([0]), 0.0, np.nan)

        # invalid scopes
        self.assertRaises(Exception, Uniform, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, Uniform, Scope([0], [1]), 0.0, 1.0)

    def test_structural_marginalization(self):

        uniform = Uniform(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(uniform, [1]) is not None)
        self.assertTrue(marginalize(uniform, [0]) is None)

    def test_accept(self):

        # discrete meta type (should reject)
        self.assertFalse(Uniform.accepts([([FeatureTypes.Continuous], Scope([0]))]))

        # Bernoulli feature type class (should reject)
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform], Scope([0]))]))

        # Bernoulli feature type instance
        self.assertTrue(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))]))

        # invalid feature type
        self.assertFalse(Uniform.accepts([([FeatureTypes.Discrete], Scope([0]))]))

        # conditional scope
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0], [1]))]))

        # scope length does not match number of types
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0, 1]))]))

        # multivariate signature
        self.assertFalse(Uniform.accepts([([FeatureTypes.Uniform(start=-1.0, end=2.0), FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0, 1]))]))

    def test_initialization_from_signatures(self):

        uniform = Uniform.from_signatures([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))])
        self.assertTrue(torch.isclose(uniform.start, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(uniform.end, torch.tensor(2.0)))

        # ----- invalid arguments -----

        # discrete meta type
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0]))])

        # Bernoulli feature type class
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Uniform], Scope([0]))])

        # invalid feature type
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Discrete], Scope([0]))])

        # conditional scope
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0], [1]))])

        # scope length does not match number of types
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous], Scope([0, 1]))])

        # multivariate signature
        self.assertRaises(ValueError, Uniform.from_signatures, [([FeatureTypes.Continuous, FeatureTypes.Continuous], Scope([0, 1]))])

    def test_autoleaf(self):

        # make sure leaf is registered
        self.assertTrue(AutoLeaf.is_registered(Uniform))

        # make sure leaf is correctly inferred
        self.assertEqual(Uniform, AutoLeaf.infer([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))]))

        # make sure AutoLeaf can return correctly instantiated object
        uniform = AutoLeaf([([FeatureTypes.Uniform(start=-1.0, end=2.0)], Scope([0]))])
        self.assertTrue(isinstance(uniform, Uniform))
        self.assertTrue(torch.isclose(uniform.start, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(uniform.end, torch.tensor(2.0)))

    def test_base_backend_conversion(self):

        start = random.random()
        end = start + 1e-7 + random.random()

        node_uniform = BaseUniform(Scope([0]), start, end)
        torch_uniform = Uniform(Scope([0]), start, end)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_uniform.get_params()]),
                np.array([*toBase(torch_uniform).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_uniform.get_params()]),
                np.array([*toTorch(node_uniform).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
