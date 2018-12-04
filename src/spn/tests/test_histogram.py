import unittest

from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.histogram.Inference import add_histogram_inference_support
from spn.structure.leaves.parametric.Parametric import *
import numpy as np


class TestParametric(unittest.TestCase):
    def setUp(self):
        add_histogram_inference_support()

    def test_Histogram_discrete_inference(self):
        data = np.array([1, 1, 2, 3, 3, 3]).reshape(-1, 1)
        ds_context = Context([MetaType.DISCRETE])
        ds_context.add_domains(data)
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
        ds_context.add_domains(data)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=True)
        # print(np.var(data.shape[0]))
        prob = np.exp(log_likelihood(hist, data))
        self.assertAlmostEqual(float(prob[0]), 3 / 9)
        self.assertAlmostEqual(float(prob[1]), 3 / 9)
        self.assertAlmostEqual(float(prob[2]), 2 / 9)
        self.assertAlmostEqual(float(prob[3]), 4 / 9)
        self.assertAlmostEqual(float(prob[4]), 4 / 9)
        self.assertAlmostEqual(float(prob[5]), 4 / 9)

    def test_spike(self):
        np.random.seed(17)
        data = np.random.normal(10, 0.01, size=200).tolist() + np.random.normal(30, 10, size=200).tolist()
        data = np.array(data).reshape((-1, 1))
        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=False, hist_source="kde")
        x = np.linspace(0, 60, 1000).tolist() + data[:, 0].tolist()
        x = np.sort(x)
        from scipy.stats import norm

        y = 0.5 * norm.pdf(x, 10, 0.01) + 0.5 * norm.pdf(x, 30, 10)
        ye = likelihood(hist, x.reshape((-1, 1)))
        error = np.sum(np.abs(ye[:, 0] - y))
        self.assertLessEqual(error, 900)
        # print(error)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(x, y, color="r", label="real", linestyle='--', dashes=(5, 1))
        # ax.plot(x, ye, color="b", label="estimated", linestyle='--', dashes=(5, 1))
        # plt.legend()
        # plt.show()

    def test_mixture_gaussians(self):
        np.random.seed(17)
        data = np.random.normal(10, 1, size=200).tolist() + np.random.normal(30, 1, size=200).tolist()
        data = np.array(data).reshape((-1, 1))
        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)
        hist = create_histogram_leaf(data, ds_context, [0], alpha=False, hist_source="kde")
        x = np.linspace(0, 60, 1000).tolist() + data[:, 0].tolist()
        x = np.sort(x)
        from scipy.stats import norm

        y = 0.5 * norm.pdf(x, 10, 1) + 0.5 * norm.pdf(x, 30, 1)
        ye = likelihood(hist, x.reshape((-1, 1)))
        error = np.sum(np.abs(ye[:, 0] - y))
        # print(error)
        self.assertLessEqual(error, 7)

    def test_valid_histogram(self):
        np.random.seed(17)
        data = [1] + [5] * 20 + [7] + [10] * 50 + [20] + [30] * 10
        data = np.array(data).reshape((-1, 1))
        ds_context = Context([MetaType.REAL])
        ds_context.add_domains(data)

        hist = create_histogram_leaf(data, ds_context, [0], alpha=False, hist_source="kde")
        self.assertGreater(len(hist.bin_repr_points), 1)


if __name__ == "__main__":
    unittest.main()
