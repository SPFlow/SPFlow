from spflow.meta.scope.scope import Scope
from spflow.base.learning.learn_spn.learn_spn import cluster_by_kmeans, partition_by_rdc, learn_spn
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.base.structure.nodes.node import SPNProductNode, SPNSumNode

import numpy as np
import unittest
import random


# dummy clustering and partition methods
def clustering_fn(x):
    # split into two approximately equal sized clusters
    mask = np.zeros(x.shape[0])
    mask[int(x.shape[0]/2):] = 1
    return mask


def partitioning_fn(x):
    ids = np.zeros(x.shape[1])

    if not partitioning_fn.alternate or partitioning_fn.partition:
        # split into two approximately equal sized partitions
        partitioning_fn.partition = False
        ids[:int(x.shape[1]/2)] = 1
    else:
        partitioning_fn.partition = True
    return ids


class TestNode(unittest.TestCase):
    def test_kmeans_clustering_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # simulate cluster data
        cluster_1 = np.random.randn(100, 1) - 5.0
        cluster_2 = np.random.randn(100, 1) + 5.0

        # compute clusters using k-means
        cluster_mask = cluster_by_kmeans(np.vstack([cluster_1, cluster_2]), n_clusters=2)

        # cluster id can either be 0 or 1
        cluster_id = cluster_mask[0] 
        # make sure all first 100 entries have the same cluster id
        self.assertTrue(np.all(cluster_mask[:100] == cluster_id))

        # second cluster id should be different from first
        cluster_id = (cluster_id + 1) % 2
        self.assertTrue(np.all(cluster_mask[100:] == cluster_id))
    
    def test_kmeans_clustering_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # simulate cluster data
        cluster_1 = np.random.randn(100, 1) - 10.0
        cluster_2 = np.random.randn(100, 1) + 10.0
        cluster_3 = np.random.randn(100, 1)

        # compute clusters using k-means
        cluster_mask = cluster_by_kmeans(np.vstack([cluster_1, cluster_2, cluster_3]), n_clusters=3)

        cluster_ids = [0,1,2]

        # cluster id can either be 0,1 or 2
        cluster_id = cluster_mask[0]
        cluster_ids.remove(cluster_id)

        # make sure all first 100 entries have the same cluster id
        self.assertTrue(np.all(cluster_mask[:100] == cluster_id))

        # second cluster id should be different from first
        cluster_id = cluster_mask[100]
        self.assertTrue(cluster_id in cluster_ids) 
        cluster_ids.remove(cluster_id)

        self.assertTrue(np.all(cluster_mask[100:200] == cluster_id))

        # third cluster id should be different from first two
        cluster_id = cluster_mask[200]
        self.assertTrue(cluster_id in cluster_ids)
        cluster_ids.remove(cluster_id)

        self.assertTrue(np.all(cluster_mask[200:] == cluster_id))
 
    def test_rdc_partitioning_1(self):
        
        # set seed
        np.random.seed(0)
        random.seed(0)

        # simulate partition data
        data_partition_1 = np.random.randn(100, 1) + 10.0
        data_partition_2 = np.random.randn(100, 1) - 10.0

        # compute clusters using k-means
        partition_mask = partition_by_rdc(np.hstack([data_partition_1, data_partition_2]), threshold=0.5)

        # should be two partitions
        self.assertTrue(len(np.unique(partition_mask)) == 2)

    def test_rdc_partitioning_2(self):
        
        # set seed
        np.random.seed(0)
        random.seed(0)

        # simulate partition data
        data_partition_1 = np.random.multivariate_normal(np.zeros(2), np.array([[1, 0.5],[0.5, 1]]), size=(100,))
        data_partition_2 = np.random.randn(100, 1) + 10.0

        # compute clusters using k-means
        partition_mask = partition_by_rdc(np.hstack([data_partition_1, data_partition_2]), threshold=0.5)

        # should be two partitions
        self.assertTrue(len(np.unique(partition_mask)) == 2)

        # check if partitions are correct (order is irrelevant)
        partition_1 = np.where(partition_mask == 0)[0]
        self.assertTrue(np.all(partition_1 == [0,1]) or np.all(partition_1 == [2]))

    def test_rdc_partitioning_nan(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # simulate partition data
        data_partition_1 = np.random.randn(100, 1) + 10.0
        data_partition_2 = np.random.randn(100, 1) - 10.0
        
        # insert invalid value
        data_partition_2[0] = np.nan

        self.assertRaises(ValueError, partition_by_rdc, np.hstack([data_partition_1, data_partition_2]), threshold=0.5)
    
    def test_learn_1(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        data = np.random.randn(100, 3)

        # ----- min_features_slice > scope size (no splitting or clustering) -----

        partitioning_fn.alternate = True
        partitioning_fn.partition = True

        spn = learn_spn(data, partitioning_method=partitioning_fn, clustering_method=clustering_fn, fit_leaves=False, min_features_slice=4)

        # check resulting graph
        self.assertTrue(isinstance(spn, SPNProductNode))
        # children of product node should be leaves since scope is originally multivariate and no partitioning/clustering occurs
        self.assertTrue(all([isinstance(child, Gaussian) for child in spn.children]))

    def test_learn_spn_2(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        data = np.random.randn(100, 3)

        # ----- min_instances_slice_100, alternate partitioning -----

        partitioning_fn.alternate = True
        partitioning_fn.partition = True

        spn = learn_spn(data, partitioning_method=partitioning_fn, clustering_method=clustering_fn, fit_leaves=False, min_instances_slice=51)

        # check resulting graph
        self.assertTrue(isinstance(spn, SPNProductNode))
        partition_1, partition_2 = spn.children
        # partition 1
        self.assertTrue(isinstance(partition_1, SPNSumNode)) 
        partition_1_clustering_1, partition_1_clustering_2 = partition_1.children
        # children of both clusterings should be product nodes since this partition is originally multivariate
        self.assertTrue(isinstance(partition_1_clustering_1, SPNProductNode))
        self.assertTrue(all([isinstance(child, Gaussian) for child in partition_1_clustering_1.children]))
        self.assertTrue(isinstance(partition_1_clustering_1, SPNProductNode))
        self.assertTrue(all([isinstance(child, Gaussian) for child in partition_1_clustering_2.children]))
        # partition 2
        self.assertTrue(isinstance(partition_2, Gaussian))

    def test_learn_spn_3(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        data = np.random.randn(100, 3)

        # ----- successive partitioning -----

        partitioning_fn.alternate = False

        spn = learn_spn(data, partitioning_method=partitioning_fn, clustering_method=clustering_fn, fit_leaves=False, min_instances_slice=101)

        # check resulting graph
        self.assertTrue(isinstance(spn, SPNProductNode))
        partition_1, partition_2 = spn.children
        # partition 1
        self.assertTrue(isinstance(partition_1, SPNProductNode)) 
        self.assertTrue(all([isinstance(child, Gaussian) for child in partition_1.children]))
        # partition 2
        self.assertTrue(isinstance(partition_2, Gaussian))
    
    def test_learn_spn_invalid_arguments(self):

        # evidence in scope
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,2), Scope([0,1],[2]))
        # scope length does not match data shape
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,3), Scope([0,1]))
        # invalid clustering method
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,3), clustering_method='invalid_option', partitioning_method='rdc')
        # invalid partitioning method
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,3), clustering_method='kmeans', partitioning_method='invalid_option')
        # invalid min number of instances for slicing
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,3), min_instances_slice=1)
        # invalid min number of features for slicing
        self.assertRaises(ValueError, learn_spn, np.random.randn(1,3), min_features_slice=1)


if __name__ == "__main__":
    unittest.main()