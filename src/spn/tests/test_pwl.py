import unittest

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Parametric import *
import numpy as np

from spn.structure.leaves.piecewise.Inference import add_piecewise_inference_support
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf


class TestPWL(unittest.TestCase):
    def test_PWL_no_variance(self):
        data = np.array([1.0, 1.0]).reshape(-1, 1)
        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)
        with self.assertRaises(AssertionError):
            create_piecewise_leaf(data, ds_context, scope=[0], hist_source="kde")

    def test_PWL(self):
        # data = np.array([1.0, 1.0, 2.0, 3.0]*100).reshape(-1, 1)

        data = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]

        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)
        leaf = create_piecewise_leaf(data, ds_context, scope=[0], prior_weight=None, hist_source="kde")
        prob = np.exp(log_likelihood(leaf, data))

        # TODO: add more test to the PWL


if __name__ == "__main__":
    unittest.main()
