from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.sampling_context import SamplingContext
from spflow.base.structure.nodes.leaves.parametric.log_normal import LogNormal
from spflow.base.sampling.nodes.leaves.parametric.log_normal import sample
from spflow.base.sampling.module import sample

import numpy as np

import unittest


class TestLogNormal(unittest.TestCase):
    def test_sampling_1(self):

        # ----- mean = 0.0, std = 1.0 -----

        log_normal = LogNormal(Scope([0]), 0.0, 1.0)

        data = np.array([[np.nan], [np.nan], [np.nan]])

        samples = sample(log_normal, data, sampling_ctx=SamplingContext([0, 2]))

        self.assertTrue(all(np.isnan(samples) == np.array([[False], [True], [False]])))

        samples = sample(log_normal, 1000)
        self.assertTrue(
            np.isclose(samples.mean(), np.exp(0.0 + (1.0 ** 2 / 2.0)), rtol=0.1)
        )
    
    def test_sampling_2(self):

        # ----- mean = 1.0, std = 0.5 -----

        log_normal = LogNormal(Scope([0]), 1.0, 0.5)

        samples = sample(log_normal, 1000)
        self.assertTrue(
            np.isclose(samples.mean(), np.exp(1.0 + (0.5 ** 2 / 2.0)), rtol=0.1)
        )


if __name__ == "__main__":
    unittest.main()