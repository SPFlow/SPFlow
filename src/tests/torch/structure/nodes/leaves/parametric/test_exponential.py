from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.exponential import (
    Exponential as BaseExponential,
)
from spflow.base.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.exponential import (
    Exponential,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.exponential import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestExponential(unittest.TestCase):
    def test_initialization(self):

        # Valid parameters for Exponential distribution: l>0

        # l > 0
        Exponential(
            Scope([0]), torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        )
        # l = 0 and l < 0
        self.assertRaises(Exception, Exponential, Scope([0]), 0.0)
        self.assertRaises(
            Exception,
            Exponential,
            Scope([0]),
            torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)),
        )
        # l = inf and l = nan
        self.assertRaises(Exception, Exponential, Scope([0]), np.inf)
        self.assertRaises(Exception, Exponential, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Exponential, Scope([]), 0.5)
        self.assertRaises(Exception, Exponential, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Exponential, Scope([0], [1]), 0.5)

    def test_structural_marginalization(self):

        exponential = Exponential(Scope([0]), 1.0)

        self.assertTrue(marginalize(exponential, [1]) is not None)
        self.assertTrue(marginalize(exponential, [0]) is None)

    def test_base_backend_conversion(self):

        l = random.random()

        torch_exponential = Exponential(Scope([0]), l)
        node_exponential = BaseExponential(Scope([0]), l)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_exponential.get_params()]),
                np.array([*toBase(torch_exponential).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_exponential.get_params()]),
                np.array([*toTorch(node_exponential).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
