import unittest
import torch
import numpy as np
from spflow.meta.data.scope import Scope
from spflow.torch.structure.layers.layer import (
    SPNSumLayer,
    SPNPartitionLayer,
    SPNHadamardLayer,
    marginalize,
)
from spflow.torch.structure.layers.leaves.parametric.gaussian import (
    GaussianLayer,
    marginalize,
)
from spflow.torch.structure.nodes.node import SPNSumNode
from spflow.torch.structure.rat.rat_spn import (
    RatSPN,
    marginalize,
    toBase,
    toTorch,
)
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.rat.rat_spn import RatSPN as BaseRatSPN
from spflow.base.structure.nodes.node import SPNSumNode as BaseSPNSumNode
from spflow.base.structure.layers.layer import SPNSumLayer as BaseSPNSumLayer
from spflow.base.structure.layers.layer import (
    SPNPartitionLayer as BaseSPNPartitionLayer,
)
from spflow.base.structure.layers.layer import (
    SPNHadamardLayer as BaseSPNHadamardLayer,
)
from spflow.base.structure.layers.leaves.parametric.gaussian import (
    GaussianLayer as BaseGaussianLayer,
)


def get_rat_spn_properties(rat_spn: RatSPN):

    n_sum_nodes = 1  # root node
    n_product_nodes = 0
    n_leaf_nodes = 0

    layers = [rat_spn.root_region]

    while layers:
        layer = layers.pop()

        # internal region
        if isinstance(layer, SPNSumLayer):
            n_sum_nodes += layer.n_out
        # partition
        elif isinstance(layer, SPNPartitionLayer):
            n_product_nodes += layer.n_out
        # multivariate leaf region
        elif isinstance(layer, SPNHadamardLayer):
            n_product_nodes += layer.n_out
        elif isinstance(layer, GaussianLayer):
            n_leaf_nodes += layer.n_out
        else:
            raise TypeError(f"Encountered unknown layer of type {type(layer)}.")

        layers += list(layer.children())

    return n_sum_nodes, n_product_nodes, n_leaf_nodes


class TestRatSpn(unittest.TestCase):
    def test_rat_spn_initialization(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1
        )

        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=0,
            n_region_nodes=1,
            n_leaf_nodes=1,
        )
        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=1,
            n_region_nodes=0,
            n_leaf_nodes=1,
        )
        self.assertRaises(
            ValueError,
            RatSPN,
            region_graph,
            n_root_nodes=1,
            n_region_nodes=1,
            n_leaf_nodes=0,
        )

    def test_rat_spn_1(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 4)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_2(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=1
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 6)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_3(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=2
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=2, n_region_nodes=2, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 23)
        self.assertEqual(n_product_nodes, 48)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_4(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=3, n_region_nodes=3, n_leaf_nodes=3
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 162)
        self.assertEqual(n_leaf_nodes, 63)

    def test_rat_spn_5(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=1, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 3)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 7)

    def test_rat_spn_6(self):

        random_variables = list(range(9))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=1, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=1, n_region_nodes=1, n_leaf_nodes=1
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 5)
        self.assertEqual(n_product_nodes, 4)
        self.assertEqual(n_leaf_nodes, 9)

    def test_rat_spn_7(self):

        random_variables = list(range(7))
        region_graph = random_region_graph(
            Scope(random_variables), depth=2, replicas=2, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=2, n_region_nodes=2, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 7)
        self.assertEqual(n_product_nodes, 40)
        self.assertEqual(n_leaf_nodes, 28)

    def test_rat_spn_8(self):

        random_variables = list(range(20))
        region_graph = random_region_graph(
            Scope(random_variables), depth=3, replicas=3, n_splits=3
        )

        rat_spn = RatSPN(
            region_graph, n_root_nodes=3, n_region_nodes=3, n_leaf_nodes=2
        )

        n_sum_nodes, n_product_nodes, n_leaf_nodes = get_rat_spn_properties(
            rat_spn
        )
        self.assertEqual(n_sum_nodes, 49)
        self.assertEqual(n_product_nodes, 267)
        self.assertEqual(n_leaf_nodes, 120)

    def test_rat_spn_backend_conversion_1(self):

        # create region graph
        region_graph = random_region_graph(
            scope=Scope(list(range(128))), depth=5, replicas=2, n_splits=2
        )

        # create torch rat spn from region graph
        torch_rat = RatSPN(
            region_graph, n_root_nodes=4, n_region_nodes=2, n_leaf_nodes=3
        )

        # change some parameters
        modules = [torch_rat.root_node]

        while modules:
            module = modules.pop()

            # modules consisting of product nodes have no parameters
            # modules consisting of sum nodes are already random
            # only need to randomize parameters of leaf layers
            if isinstance(module, GaussianLayer):
                module.set_params(
                    mean=torch.randn(module.mean.shape),
                    std=torch.rand(module.std.shape) + 1e-8,
                )

            modules += list(module.children())

        base_rat = toBase(torch_rat)

        modules = [(torch_rat.root_node, base_rat.root_node)]

        while modules:

            torch_module, base_module = modules.pop()

            if isinstance(torch_module, SPNSumNode):
                if not isinstance(base_module, BaseSPNSumNode):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNSumLayer):
                if not isinstance(base_module, BaseSPNSumLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNPartitionLayer):
                if not isinstance(base_module, BaseSPNPartitionLayer):
                    raise TypeError()
            if isinstance(torch_module, SPNHadamardLayer):
                if not isinstance(base_module, BaseSPNHadamardLayer):
                    raise TypeError()
            if isinstance(torch_module, GaussianLayer):
                if not isinstance(base_module, BaseGaussianLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.mean, torch_module.mean)
                )
                self.assertTrue(
                    torch.allclose(torch_module.std, torch_module.std)
                )

            modules += list(zip(torch_module.children(), base_module.children))

    def test_rat_spn_backend_conversion_2(self):

        # create region graph
        region_graph = random_region_graph(
            scope=Scope(list(range(128))), depth=5, replicas=2, n_splits=2
        )

        # create torch rat spn from region graph
        base_rat = BaseRatSPN(
            region_graph, n_root_nodes=4, n_region_nodes=2, n_leaf_nodes=3
        )

        # change some parameters
        modules = [base_rat.root_node]

        while modules:
            module = modules.pop()

            # modules consisting of product nodes have no parameters
            # modules consisting of sum nodes are already random
            # only need to randomize parameters of leaf layers
            if isinstance(module, BaseGaussianLayer):
                module.set_params(
                    mean=np.random.randn(*module.mean.shape),
                    std=np.random.rand(*module.std.shape) + 1e-8,
                )

            modules += module.children

        torch_rat = toTorch(base_rat)

        modules = [(torch_rat.root_node, base_rat.root_node)]

        while modules:

            torch_module, base_module = modules.pop()

            if isinstance(torch_module, SPNSumNode):
                if not isinstance(base_module, BaseSPNSumNode):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNSumLayer):
                if not isinstance(base_module, BaseSPNSumLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.weights, torch_module.weights)
                )
            if isinstance(torch_module, SPNPartitionLayer):
                if not isinstance(base_module, BaseSPNPartitionLayer):
                    raise TypeError()
            if isinstance(torch_module, SPNHadamardLayer):
                if not isinstance(base_module, BaseSPNHadamardLayer):
                    raise TypeError()
            if isinstance(torch_module, GaussianLayer):
                if not isinstance(base_module, BaseGaussianLayer):
                    raise TypeError()
                self.assertTrue(
                    torch.allclose(torch_module.mean, torch_module.mean)
                )
                self.assertTrue(
                    torch.allclose(torch_module.std, torch_module.std)
                )

            modules += list(zip(torch_module.children(), base_module.children))


if __name__ == "__main__":
    unittest.main()
