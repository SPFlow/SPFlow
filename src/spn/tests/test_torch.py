from typing import Any, List, Set, Tuple
from spn.structure.graph.Node import Node, ProductNode, SumNode, LeafNode
from spn.backend.pytorch.Node import TorchNode, TorchProductNode, TorchSumNode, TorchLeafNode, toTorch, toNodes
import unittest

class TestTorch(unittest.TestCase):
    def test_graph_conversion(self):

        # Node graph
        graph = SumNode([
                ProductNode([
                    LeafNode(scope=[1]),
                    LeafNode(scope=[2])
                ], scope=[1,2]),
                ProductNode([
                    LeafNode(scope=[3])
                ], scope=[3])
            ], scope=[1,2,3], weights=[0.3, 0.7])
        
        # conversion to PyTorch graph
        graph_torch = toTorch(graph)

        # conversion back to Node representation
        graph_nodes = toNodes(graph_torch)

        # check whether converted graph matches original graph
        self.assertEqual(graph, graph_nodes)

if __name__ == "__main__":
    unittest.main()