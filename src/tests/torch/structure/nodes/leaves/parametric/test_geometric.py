from spflow.meta.data.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.geometric import (
    Geometric as BaseGeometric,
)
from spflow.base.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.torch.structure.nodes.leaves.parametric.geometric import (
    Geometric,
    toBase,
    toTorch,
)
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.geometric import (
    log_likelihood,
)
from spflow.torch.inference.module import likelihood

import torch
import numpy as np

import random
import unittest


class TestGeometric(unittest.TestCase):
    def test_initialiation(self):

        # Valid parameters for Geometric distribution: p in (0,1]

        # p = 0
        self.assertRaises(Exception, Geometric, Scope([0]), 0.0)

        # p = inf and p = nan
        self.assertRaises(Exception, Geometric, Scope([0]), np.inf)
        self.assertRaises(Exception, Geometric, Scope([0]), np.nan)

        # invalid scopes
        self.assertRaises(Exception, Geometric, Scope([]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0, 1]), 0.5)
        self.assertRaises(Exception, Geometric, Scope([0], [1]), 0.5)

    def test_structural_marginalization(self):

        geometric = Geometric(Scope([0]), 0.5)

        self.assertTrue(marginalize(geometric, [1]) is not None)
        self.assertTrue(marginalize(geometric, [0]) is None)

    def test_base_backend_conversion(self):

        p = random.random()

        torch_geometric = Geometric(Scope([0]), p)
        node_geometric = BaseGeometric(Scope([0]), p)

        # check conversion from torch to python
        self.assertTrue(
            np.allclose(
                np.array([*torch_geometric.get_params()]),
                np.array([*toBase(torch_geometric).get_params()]),
            )
        )
        # check conversion from python to torch
        self.assertTrue(
            np.allclose(
                np.array([*node_geometric.get_params()]),
                np.array([*toTorch(node_geometric).get_params()]),
            )
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
