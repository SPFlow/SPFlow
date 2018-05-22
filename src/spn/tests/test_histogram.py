import unittest

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf, add_domains
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import *


class TestParametric(unittest.TestCase):
    def setUp(self):
        add_histogram_inference_support()

    def test_Histogram_discrete_inference(self):
        data = np.array([1, 1, 2, 3, 3, 3]).reshape(-1, 1)
        ds_context = Context([MetaType.DISCRETE])
        add_domains(data, ds_context)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=False)
        prob = np.exp(log_likelihood(hist, data))

        self.assertAlmostEqual(float(prob[0]), 2 / 6)
        self.assertAlmostEqual(float(prob[1]), 2 / 6)
        self.assertAlmostEqual(float(prob[2]), 1 / 6)
        self.assertAlmostEqual(float(prob[3]), 3 / 6)
        self.assertAlmostEqual(float(prob[4]), 3 / 6)
        self.assertAlmostEqual(float(prob[5]), 3 / 6)

        data = np.array([1, 1, 2, 3, 3, 3]).reshape(-1, 1)
        ds_context = Context([MetaType.DISCRETE])
        add_domains(data, ds_context)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=1)
        prob = np.exp(log_likelihood(hist, data))

        self.assertAlmostEqual(float(prob[0]), 3 / 9)
        self.assertAlmostEqual(float(prob[1]), 3 / 9)
        self.assertAlmostEqual(float(prob[2]), 2 / 9)
        self.assertAlmostEqual(float(prob[3]), 4 / 9)
        self.assertAlmostEqual(float(prob[4]), 4 / 9)
        self.assertAlmostEqual(float(prob[5]), 4 / 9)





if __name__ == '__main__':
    unittest.main()
