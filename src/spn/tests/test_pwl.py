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


class TestParametric(unittest.TestCase):
    def setUp(self):
        add_histogram_inference_support()
        add_piecewise_inference_support()

    def test_PWL_no_variance(self):
        data = np.array([1.0, 1.0]).reshape(-1, 1)
        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)
        leaf = create_piecewise_leaf(data, ds_context, scope=[0], hist_source="kde")
        prob = np.exp(log_likelihood(leaf, data))

        self.assertAlmostEqual(float(prob[0]), 2 / 6)
        self.assertAlmostEqual(float(prob[1]), 2 / 6)



if __name__ == '__main__':
    unittest.main()
