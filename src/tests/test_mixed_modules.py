import unittest
import numpy as np
from spflow.base.structure.nodes.node_module import (
    SumNode,
    ProductNode,
    LeafNode,
)
from spflow.base.structure.rat.rat_spn import RatSpn
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.module import _get_node_counts
from spflow.base.structure.network_type import SPN
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.nodes.leaves.parametric import Gaussian


class TestMixedModules(unittest.TestCase):
    def test_spn_graph_small_with_rat_spn(self):
        region_graph = random_region_graph(X=set(range(0, 2)), depth=1, replicas=1)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * (len(region_graph.root_region.random_variables) + 1)
        )
        rat_spn_module = RatSpn(region_graph, 1, 1, 1, context)
        _isvalid_spn(rat_spn_module)

        leaf1 = rat_spn_module
        leaf2 = LeafNode(scope=[2], network_type=SPN(), context=context)
        prod1 = ProductNode(children=[leaf1, leaf2], scope=[0, 1, 2], network_type=SPN())
        prod2 = ProductNode(children=[leaf1, leaf2], scope=[0, 1, 2], network_type=SPN())
        sum = SumNode(
            children=[prod1, prod2],
            scope=[0, 1, 2],
            weights=np.array([0.3, 0.7]),
            network_type=SPN(),
        )

        _isvalid_spn(sum)
        sum_nodes, prod_nodes, leaf_nodes = _get_node_counts(sum)
        self.assertEqual(sum_nodes, 3)
        self.assertEqual(prod_nodes, 3)
        self.assertEqual(leaf_nodes, 3)


if __name__ == "__main__":
    unittest.main()
