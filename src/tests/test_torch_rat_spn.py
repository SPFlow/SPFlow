import unittest

import random
from unittest.signals import registerResult
from spflow.base.structure.nodes.node import get_nodes_by_type, ISumNode, ILeafNode
from spflow.base.inference.rat.rat_spn import log_likelihood, likelihood
import torch
import numpy as np
from spflow.base.structure.rat.region_graph import (
    random_region_graph,
    _print_region_graph,
    RegionGraph,
)
from spflow.base.structure.rat import RatSpn, construct_spn
from spflow.torch.structure.rat import (
    TorchRatSpn,
    toNodes,
    toTorch,
    _RegionLayer,
    _LeafLayer,
)
from spflow.torch.inference import log_likelihood, likelihood


class TestTorchRatSpn(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        torch.set_default_dtype(torch.float64)

    @classmethod
    def teardown_class(cls):
        torch.set_default_dtype(torch.float32)

    def test_torch_rat_spn_initialization(self):

        # create region graph
        rg = RegionGraph()

        self.assertRaises(
            ValueError,
            TorchRatSpn,
            rg,
            num_nodes_root=0,
            num_nodes_region=1,
            num_nodes_leaf=1,
        )
        self.assertRaises(
            ValueError,
            TorchRatSpn,
            rg,
            num_nodes_root=1,
            num_nodes_region=0,
            num_nodes_leaf=1,
        )
        self.assertRaises(
            ValueError,
            TorchRatSpn,
            rg,
            num_nodes_root=1,
            num_nodes_region=1,
            num_nodes_leaf=0,
        )

    def test_torch_rat_spn_to_nodes(self):

        # create region graph
        rg = random_region_graph(X=set(range(1024)), depth=5, replicas=2, num_splits=4)

        # create torch rat spn from region graph
        torch_rat = TorchRatSpn(
            rg, num_nodes_root=4, num_nodes_region=2, num_nodes_leaf=3
        )

        # randomly change parameters from inital values
        for region in torch_rat.region_graph.regions:

            region_layer: _RegionLayer = torch_rat.rg_layers[region]

            if isinstance(region_layer, _LeafLayer):
                for leaf_node in region_layer.leaf_nodes:
                    # TODO: only works for Gaussians (provide method to randomize parameters ?)
                    leaf_node.set_params(random.random(), random.random() + 1e-08)

        # convert torch rat spn to node rat spn
        rat = toNodes(torch_rat)

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 1024)

        # compute outputs for node rat spn
        nodes_output = log_likelihood(rat, dummy_data)

        # compute outputs for torch rat spn
        torch_output = log_likelihood(torch_rat, torch.tensor(dummy_data))

        # compare outputs
        self.assertTrue(
            np.allclose(
                nodes_output, torch_output.detach().cpu().numpy(), equal_nan=True
            )
        )

    def test_nodes_rat_spn_to_torch(self):

        # create region graph
        rg = random_region_graph(X=set(range(1024)), depth=5, replicas=2, num_splits=4)

        # create torch rat spn from region graph
        rat = RatSpn(rg, num_nodes_root=4, num_nodes_region=2, num_nodes_leaf=3)

        sum_nodes = get_nodes_by_type(rat.output_nodes[0], ISumNode)
        leaf_nodes = get_nodes_by_type(rat.output_nodes[0], ILeafNode)

        # randomly change parameters from inital values
        for node in leaf_nodes:
            node.set_params(random.random(), random.random() + 1e-08)

        # convert node rat spn to torch rat spn
        torch_rat = toTorch(rat)

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 1024)

        # compute outputs for node rat spn
        nodes_output = log_likelihood(rat, dummy_data)

        # compute outputs for torch rat spn
        torch_output = log_likelihood(torch_rat, torch.tensor(dummy_data))

        # compare outputs
        self.assertTrue(
            np.allclose(
                nodes_output, torch_output.detach().cpu().numpy(), equal_nan=True
            )
        )

    def test_torch_rat_spn_to_nodes_to_torch(self):

        # create region graph
        rg = random_region_graph(X=set(range(1024)), depth=5, replicas=2, num_splits=4)

        # create torch rat spn from region graph
        torch_rat = TorchRatSpn(
            rg, num_nodes_root=4, num_nodes_region=2, num_nodes_leaf=3
        )

        # randomly change parameters from inital values
        for region in torch_rat.region_graph.regions:

            region_layer: _RegionLayer = torch_rat.rg_layers[region]

            if isinstance(region_layer, _LeafLayer):
                for leaf_node in region_layer.leaf_nodes:
                    # TODO: only works for Gaussians (provide method to randomize parameters ?)
                    leaf_node.set_params(random.random(), random.random() + 1e-08)

        # convert torch rat spn to nodes and back to torch
        torch_rat_2 = toTorch(toNodes(torch_rat))

        self.assertTrue(
            set(torch_rat.rg_layers.keys()) == set(torch_rat_2.rg_layers.keys())
        )

        for p1, p2 in [
            (torch_rat.rg_layers[k], torch_rat_2.rg_layers[k])
            for k in torch_rat.rg_layers.keys()
        ]:

            if isinstance(p1, _RegionLayer):
                self.assertTrue(torch.allclose(p1.weights, p2.weights))
            elif isinstance(p1, _LeafLayer):
                for l1, l2 in zip(p1.leaf_nodes, p2.leaf_nodes):

                    self.assertTrue(l1.scope == l2.scope)

                    for l1_p, l2_p in zip(l1.get_params(), l2.get_params()):
                        self.assertTrue(
                            torch.allclose(torch.tensor(l1_p), torch.tensor(l2_p))
                        )

        # create dummy input data (batch size x random variables)
        dummy_data = np.random.randn(3, 1024)

        # compute outputs for node rat spn
        torch_output = log_likelihood(torch_rat, torch.tensor(dummy_data))

        # compute outputs for torch rat spn
        torch_output_2 = log_likelihood(torch_rat_2, torch.tensor(dummy_data))

        # compare outputs
        self.assertTrue(torch.allclose(torch_output.exp(), torch_output_2.exp()))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    unittest.main()
