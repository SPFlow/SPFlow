from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.uniform import Uniform
from spflow.base.sampling.nodes.leaves.parametric.uniform import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestUniform(unittest.TestCase):
    def test_sampling(self):

        # ----- a = -1.0, b = 2.5 -----

        uniform = Uniform(Scope([0]), -1.0, 2.5)
        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(uniform, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))

        samples = sample(uniform, 1000)
        self.assertTrue(all((samples >= -1.0) & (samples <= 2.5)))


if __name__ == "__main__":
    unittest.main()