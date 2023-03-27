import unittest

from spn.structure.Base import get_nodes_by_type
from spn.algorithms.EM import EM_optimization
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import numpy as np
from sklearn.datasets import make_blobs

from spn.structure.leaves.parametric.Parametric import Gaussian


class TestEM(unittest.TestCase):
    def test_optimization(self):
        np.random.seed(17)
        d1 = np.random.normal(10, 1, size=4000).tolist()
        d2 = np.random.normal(30, 1, size=4000).tolist()
        data = d1 + d2
        data = np.array(data).reshape((-1, 4))
        data = data.astype(np.float32)

        ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1], parametric_types=[Gaussian] * data.shape[1])
        ds_context.add_domains(data)

        spn = learn_parametric(data, ds_context)

        spn.weights = [0.8, 0.2]
        spn.children[0].children[0].mean = 3.0

        py_ll = np.sum(log_likelihood(spn, data))

        print(spn.weights, spn.children[0].children[0].mean)

        EM_optimization(spn, data, iterations=1000)

        print(spn.weights, spn.children[0].children[0].mean)

        py_ll_opt = np.sum(log_likelihood(spn, data))

        self.assertLessEqual(py_ll, py_ll_opt)
        self.assertAlmostEqual(spn.weights[0], 0.5, 4)
        self.assertAlmostEqual(spn.weights[1], 0.5, 4)

        c1_mean = spn.children[0].children[0].mean
        c2_mean = spn.children[1].children[0].mean
        self.assertEqual(round(min(c1_mean, c2_mean)), 10)
        self.assertEqual(round(max(c1_mean, c2_mean)), 30)

    def test_clustering(self):
        np.random.seed(0)

        centers = [[10, 10], [-10, -10], [10, -10]]
        center_stdev = 0.7
        X, labels_true = make_blobs(n_samples=1000000, centers=centers, cluster_std=center_stdev)

        initial_cluster_centers = [[1, 1], [0, 0], [1, 0]]
        g0x = Gaussian(mean=initial_cluster_centers[0][0], stdev=1.0, scope=0)
        g0y = Gaussian(mean=initial_cluster_centers[0][1], stdev=1.0, scope=1)
        g1x = Gaussian(mean=initial_cluster_centers[1][0], stdev=1.0, scope=0)
        g1y = Gaussian(mean=initial_cluster_centers[1][1], stdev=1.0, scope=1)
        g2x = Gaussian(mean=initial_cluster_centers[2][0], stdev=1.0, scope=0)
        g2y = Gaussian(mean=initial_cluster_centers[2][1], stdev=1.0, scope=1)

        spn = 0.6 * (0.5 * (g0x * g0y) + 0.5 * (g1x * g1y)) + 0.4 * (g2x * g2y)

        EM_optimization(spn, X, iterations=5)

        cluster_centers2 = [[g0x.mean, g0y.mean], [g1x.mean, g1y.mean], [g2x.mean, g2y.mean]]

        print("\ntrue centers", centers)
        print("initial ctrs", initial_cluster_centers)
        print("final   ctrs", cluster_centers2)

        for i, cluster_location in enumerate(centers):
            self.assertAlmostEqual(cluster_location[0], cluster_centers2[i][0], 2)
            self.assertAlmostEqual(cluster_location[1], cluster_centers2[i][1], 2)

        for n in get_nodes_by_type(spn, Gaussian):
            self.assertAlmostEqual(n.stdev, center_stdev, 2)


if __name__ == "__main__":
    unittest.main()
