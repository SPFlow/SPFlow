from spflow.torch.structure.layers.layer import SPNSumLayer, SPNProductLayer, SPNPartitionLayer, SPNHadamardLayer, marginalize, toBase, toTorch
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian, toBase, toTorch
from spflow.base.structure.layers.layer import SPNSumLayer as BaseSPNSumLayer
from spflow.base.structure.layers.layer import SPNProductLayer as BaseSPNProductLayer
from spflow.base.structure.layers.layer import SPNPartitionLayer as BaseSPNPartitionLayer
from spflow.base.structure.layers.layer import SPNHadamardLayer as BaseSPNHadamardLayer
from spflow.base.structure.nodes.leaves.parametric.gaussian import Gaussian as BaseGaussian
from spflow.meta.scope.scope import Scope
from ..nodes.dummy_node import DummyNode
import torch
import numpy as np
import unittest
import itertools


class TestNode(unittest.TestCase):
    def test_sum_layer_initialization(self):

        # dummy children over same scope
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0,1]))]

        # ----- check attributes after correct initialization -----

        l = SPNSumLayer(n_nodes=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1]), Scope([0,1]), Scope([0,1])]))
        # make sure weight property works correctly
        self.assertTrue(l.weights.shape == (3, 3))

        # ----- same weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4]])

        # two dimensional weight array
        l = SPNSumLayer(n_nodes=3, children=input_nodes, weights=weights)

        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights))

        # one dimensional weight array
        l = SPNSumLayer(n_nodes=3, children=input_nodes, weights=weights.squeeze(0))

        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights))

        # ----- different weights for all nodes -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]])
        
        l = SPNSumLayer(n_nodes=3, children=input_nodes, weights=weights)
        for i in range(3):
            self.assertTrue(torch.allclose(l.weights[i], weights[i]))

        # ----- two dimensional weight array of wrong shape -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])

        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights.T)

        # ----- weights not summing up to one per row -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.7, 0.3], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)

        # ----- non-positive weights -----
        weights = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.0, 0.5], [0.1, 0.7, 0.2]])
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes, weights)

        # ----- children of different scopes -----
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0]))]
        self.assertRaises(ValueError, SPNSumLayer, 3, input_nodes)
    
        # ----- no children -----
        self.assertRaises(ValueError, SPNSumLayer, 3, [])

    def test_sum_layer_structural_marginalization(self):
        
        # dummy children over same scope
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([0,1])), DummyNode(Scope([0,1]))]
        l = SPNSumLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [0])
        self.assertTrue(l_marg.scopes_out == [Scope([1]), Scope([1]), Scope([1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1]), Scope([0,1]), Scope([0,1])])
        self.assertTrue(torch.allclose(l.weights, l_marg.weights))
    
    def test_sum_layer_backend_conversion_1(self):

        torch_sum_layer = SPNSumLayer(n_nodes=3, children=[Gaussian(Scope([0])), Gaussian(Scope([0])), Gaussian(Scope([0]))])

        base_sum_layer = toBase(torch_sum_layer)
        self.assertTrue(np.allclose(base_sum_layer.weights, torch_sum_layer.weights.detach().numpy()))
        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)
    
    def test_sum_layer_backend_conversion_2(self):

        base_sum_layer = BaseSPNSumLayer(n_nodes=3, children=[BaseGaussian(Scope([0])), BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))])
        
        torch_sum_layer = toTorch(base_sum_layer)
        self.assertTrue(np.allclose(base_sum_layer.weights, torch_sum_layer.weights.detach().numpy()))
        self.assertEqual(base_sum_layer.n_out, torch_sum_layer.n_out)

    def test_product_layer_initialization(self):
        
        # dummy children pair-wise disjoint scopes
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([2]))]

        # ----- check attributes after correct initialization -----

        l = SPNProductLayer(n_nodes=3, children=input_nodes)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1,2,3]), Scope([0,1,2,3]), Scope([0,1,2,3])]))
        
        # ----- children of non-pair-wise disjoint scopes -----
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([1]))]
        self.assertRaises(ValueError, SPNProductLayer, 3, input_nodes)

        # ----- no children -----
        self.assertRaises(ValueError, SPNProductLayer, 3, [])

    def test_product_layer_structural_marginalization(self):
        
        # dummy children over pair-wise disjoint scopes
        input_nodes = [DummyNode(Scope([0,1])), DummyNode(Scope([3])), DummyNode(Scope([2]))]
        l = SPNProductLayer(n_nodes=3, children=input_nodes)

        # ----- marginalize over entire scope -----
        self.assertTrue(marginalize(l, [0,1,2,3]) == None)

        # ----- marginalize over partial scope -----
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1,2]), Scope([0,1,2]), Scope([0,1,2])])
        # number of children should be reduced by one (i.e., marginalized over)
        self.assertTrue(len(list(l_marg.children())) == 2)

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1,2,3]), Scope([0,1,2,3]), Scope([0,1,2,3])])

        # ----- pruning -----
        l = SPNProductLayer(n_nodes=3, children=input_nodes[:2])

        l_marg = marginalize(l, [0,1], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))

    def test_product_layer_backend_conversion_1(self):

        torch_product_layer = SPNProductLayer(n_nodes=3, children=[Gaussian(Scope([0])), Gaussian(Scope([1])), Gaussian(Scope([2]))])

        base_product_layer = toBase(torch_product_layer)
        self.assertEqual(base_product_layer.n_out, torch_product_layer.n_out)
    
    def test_product_layer_backend_conversion_2(self):

        base_product_layer = BaseSPNProductLayer(n_nodes=3, children=[BaseGaussian(Scope([0])), BaseGaussian(Scope([1])), BaseGaussian(Scope([2]))])
        
        torch_product_layer = toTorch(base_product_layer)
        self.assertEqual(base_product_layer.n_out, torch_product_layer.n_out)
    
    def test_partition_layer_initialization(self):
        
        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [DummyNode(Scope([1,3])), DummyNode(Scope([1,3])), DummyNode(Scope([1,3]))],
            [DummyNode(Scope([2]))]
        ]

        # ----- check attributes after correct initialization -----

        l = SPNPartitionLayer(child_partitions=input_partitions)
        # make sure number of creates nodes is correct
        self.assertEqual(l.n_out, np.prod([len(partition) for partition in input_partitions]))
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1,2,3]) for _ in range(l.n_out)]))
        # make sure order of nodes is correct (important)
        for indices, indices_torch in zip(itertools.product([0,1],[2,3,4],[5]), torch.cartesian_prod(torch.tensor([0,1]),torch.tensor([2,3,4]),torch.tensor([5]))):
            self.assertTrue(torch.all(torch.tensor(indices) == indices_torch))

        # ----- no child partitions -----
        self.assertRaises(ValueError, SPNPartitionLayer, [])

        # ----- empty partition -----
        self.assertRaises(ValueError, SPNPartitionLayer, [[]])

        # ----- scopes inside partition differ -----
        self.assertRaises(ValueError, SPNPartitionLayer, [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([1])), DummyNode(Scope([2]))]
        ])

        # ----- partitions of non-pair-wise disjoint scopes -----
        self.assertRaises(ValueError, SPNPartitionLayer, [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([0])), DummyNode(Scope([0]))]
        ])

    def test_partition_layer_structural_marginalization(self):
        
        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [DummyNode(Scope([1,3])), DummyNode(Scope([1,3])), DummyNode(Scope([1,3]))],
            [DummyNode(Scope([2]))]
        ]

        l = SPNPartitionLayer(child_partitions=input_partitions)
        # should marginalize entire module
        l_marg = marginalize(l, [0,1,2,3])
        self.assertTrue(l_marg is None)
        # should marginalize entire partition
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scope == Scope([0,1,3]))
        # should partially marginalize one partition
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scope == Scope([0,1,2]))

        # ----- pruning -----
        l = SPNPartitionLayer(child_partitions=input_partitions[1:])

        l_marg = marginalize(l, [1,3], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))

    def test_partition_layer_backend_conversion_1(self):
        
        torch_partition_layer = SPNPartitionLayer(child_partitions=[[Gaussian(Scope([0])), Gaussian(Scope([0]))], [Gaussian(Scope([1]))], [Gaussian(Scope([2])), Gaussian(Scope([2])), Gaussian(Scope([2]))]])

        base_partition_layer = toBase(torch_partition_layer)
        self.assertEqual(base_partition_layer.n_out, torch_partition_layer.n_out)

    def test_partition_layer_backend_conversion_2(self):
        
        base_partition_layer = BaseSPNPartitionLayer(child_partitions=[[BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))], [BaseGaussian(Scope([1]))], [BaseGaussian(Scope([2])), BaseGaussian(Scope([2])), BaseGaussian(Scope([2]))]])

        torch_partition_layer = toTorch(base_partition_layer)
        self.assertEqual(base_partition_layer.n_out, torch_partition_layer.n_out)

    def test_hadamard_layer_initialization(self):
    
        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([1,3])), DummyNode(Scope([1,3])), DummyNode(Scope([1,3]))],
            [DummyNode(Scope([2]))],
            [DummyNode(Scope([4])), DummyNode(Scope([4])), DummyNode(Scope([4]))]
        ]

        # ----- check attributes after correct initialization -----

        l = SPNHadamardLayer(child_partitions=input_partitions)
        # make sure number of creates nodes is correct
        self.assertEqual(l.n_out, 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0,1,2,3,4]) for _ in range(l.n_out)]))

        # only one partition
        l = SPNHadamardLayer(child_partitions=[[DummyNode(Scope([1,3])), DummyNode(Scope([1,3])), DummyNode(Scope([1,3]))]])
        # make sure number of creates nodes is correct
        self.assertEqual(l.n_out, 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1,3]) for _ in range(l.n_out)]))

        # ----- no child partitions -----
        self.assertRaises(ValueError, SPNHadamardLayer, [])

        # ----- empty partition -----
        self.assertRaises(ValueError, SPNHadamardLayer, [[]])

        # ----- scopes inside partition differ -----
        self.assertRaises(ValueError, SPNHadamardLayer, [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([1])), DummyNode(Scope([2]))]
        ])

        # ----- partitions of non-pair-wise disjoint scopes -----
        self.assertRaises(ValueError, SPNHadamardLayer, [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([0])), DummyNode(Scope([0]))]
        ])

        # ----- invalid total outputs of partitions -----
        self.assertRaises(ValueError, SPNHadamardLayer, [
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [DummyNode(Scope([1])), DummyNode(Scope([1])), DummyNode(Scope([1]))]
        ])

    def test_hadamard_layer_structural_marginalization(self):
        
        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [DummyNode(Scope([1,3])), DummyNode(Scope([1,3])), DummyNode(Scope([1,3]))],
            [DummyNode(Scope([2]))],
            [DummyNode(Scope([4])), DummyNode(Scope([4])), DummyNode(Scope([4]))]
        ]

        l = SPNHadamardLayer(child_partitions=input_partitions)
        # should marginalize entire module
        l_marg = marginalize(l, [0,1,2,3,4])
        self.assertTrue(l_marg is None)
        # should marginalize entire partition
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scope == Scope([0,1,3,4]))
        # should partially marginalize one partition
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scope == Scope([0,1,2,4]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [5])
        self.assertTrue(l_marg.scopes_out == [Scope([0,1,2,3,4]), Scope([0,1,2,3,4]), Scope([0,1,2,3,4])])

        # ----- pruning -----
        l = SPNHadamardLayer(child_partitions=input_partitions[:2])

        l_marg = marginalize(l, [1,3], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))

    def test_hadamard_layer_backend_conversion_1(self):
        
        torch_hadamard_layer = SPNHadamardLayer(child_partitions=[[Gaussian(Scope([0])), Gaussian(Scope([0]))], [Gaussian(Scope([1]))], [Gaussian(Scope([2])), Gaussian(Scope([2]))]])

        base_hadamard_layer = toBase(torch_hadamard_layer)
        self.assertEqual(base_hadamard_layer.n_out, torch_hadamard_layer.n_out)

    def test_hadamard_layer_backend_conversion_2(self):

        base_hadamard_layer = BaseSPNHadamardLayer(child_partitions=[[BaseGaussian(Scope([0])), BaseGaussian(Scope([0]))], [BaseGaussian(Scope([1]))], [BaseGaussian(Scope([2])), BaseGaussian(Scope([2]))]])

        torch_hadamard_layer = toTorch(base_hadamard_layer)
        self.assertEqual(base_hadamard_layer.n_out, torch_hadamard_layer.n_out)


if __name__ == "__main__":
    unittest.main()