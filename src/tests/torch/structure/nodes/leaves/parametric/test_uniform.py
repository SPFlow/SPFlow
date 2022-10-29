# from spflow.base.sampling.sampling_context import SamplingContext
from spflow.meta.scope.scope import Scope
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

# from spflow.torch.sampling import sample

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
