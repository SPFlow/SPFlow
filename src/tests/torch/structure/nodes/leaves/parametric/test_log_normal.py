#from spflow.base.sampling.sampling_context import SamplingContext
from spflow.meta.scope.scope import Scope
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal as BaseLogNormal
from spflow.base.inference.nodes.leaves.parametric.log_normal import log_likelihood
from spflow.torch.structure.nodes.leaves.parametric.log_normal import LogNormal, toBase, toTorch
from spflow.torch.structure.nodes.node import marginalize
from spflow.torch.inference.nodes.leaves.parametric.log_normal import log_likelihood
from spflow.torch.inference.module import likelihood
#from spflow.torch.sampling import sample

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
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nextafter(0.0, -1.0))
        # std = +-inf and std = nan
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, -np.inf)
        self.assertRaises(Exception, LogNormal, Scope([0]), mean, np.nan)

        # invalid scopes
        self.assertRaises(Exception, LogNormal, Scope([]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0, 1]), 0.0, 1.0)
        self.assertRaises(Exception, LogNormal, Scope([0],[1]), 0.0, 1.0)
    
    def test_structural_marginalization(self):

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        self.assertTrue(marginalize(log_normal, [1]) is not None)
        self.assertTrue(marginalize(log_normal, [0]) is None)

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
