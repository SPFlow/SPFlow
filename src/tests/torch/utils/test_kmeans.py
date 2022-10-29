from spflow.torch.utils.kmeans import kmeans
from sklearn.cluster import KMeans

import torch
import numpy as np
import unittest
import random


class TestNode(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_kmeans_initialization(self):

        # invalid number of clusters
        self.assertRaises(ValueError, kmeans, torch.randn((3, 4)), n_clusters=0)
        # more clusters than data points
        self.assertRaises(ValueError, kmeans, torch.randn((3, 4)), n_clusters=4)

        # invalid centroid initialization strategy
        self.assertRaises(
            ValueError,
            kmeans,
            torch.randn((3, 4)),
            n_clusters=2,
            centroids="invalid_init_strategy",
        )
        # kmeans++ initialization
        kmeans(torch.randn(3, 4), n_clusters=2, centroids="kmeans++")
        # random initialization
        kmeans(torch.randn(3, 4), n_clusters=2, centroids="random")

        # specified number of centroids does not match number of desired clusters
        self.assertRaises(
            ValueError,
            kmeans,
            torch.randn((3, 4)),
            n_clusters=3,
            centroids=torch.randn((2, 4)),
        )
        # number of feature dimensions for specified centroids does not match data
        self.assertRaises(
            ValueError,
            kmeans,
            torch.randn((3, 4)),
            n_clusters=3,
            centroids=torch.randn((3, 3)),
        )
        # valid centroids
        kmeans(torch.randn(3, 4), n_clusters=2, centroids=torch.randn(2, 4))

        # 1-dimensional centroids
        kmeans(torch.randn(3, 1), n_clusters=2, centroids=torch.rand(2))

    def test_kmeans_clustering(self):

        # set seed
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        data = np.vstack(
            [
                np.random.multivariate_normal(
                    np.array([-10.0, -5]), np.eye(2), (300,)
                ),
                np.random.multivariate_normal(
                    np.array([0.0, 0.0]), np.eye(2), (300,)
                ),
                np.random.multivariate_normal(
                    np.array([10.0, 5.0]), np.eye(2), (300,)
                ),
            ]
        )

        starting_centroids = np.vstack([data[0], data[300], data[600]])

        # perform sklearn's k-means clustering
        k_means = KMeans(
            n_clusters=3, init=starting_centroids, n_init=1, random_state=0
        )
        np_labels = k_means.fit_predict(data)
        np_centroids = k_means.cluster_centers_

        # perform our k-means clustering in torch
        torch_centroids, torch_labels = kmeans(
            torch.tensor(data),
            n_clusters=3,
            centroids=torch.tensor(starting_centroids),
        )

        self.assertTrue(np.allclose(np_centroids, torch_centroids.numpy()))
        self.assertTrue(np.all(np_labels == torch_labels.numpy()))


if __name__ == "__main__":
    unittest.main()
