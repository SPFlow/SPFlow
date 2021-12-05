import unittest
from spflow.torch.inference import log_likelihood, likelihood
from spflow.torch.structure.nodes import TorchProductNode, TorchSumNode, TorchGaussian
import sys
import torch
from spflow.base.inference.module import likelihood, log_likelihood
from spflow.base.structure.nodes.leaves.parametric import Gaussian
import numpy as np
from spflow.base.structure.network_type import SPN
from spflow.base.inference.nodes import likelihood, log_likelihood
from spflow.base.structure.nodes import ISumNode, IProductNode
from spflow.base.learning.context import RandomVariableContext  # type: ignore
from spflow.base.structure.rat.rat_spn import RatSpn
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.nodes.node import _get_node_counts


class TestMemoization(unittest.TestCase):
    def test_memoization_torch_log_ll(self):

        #       [A]         (sum node)
        #       / \
        #      /   \
        #    [B]   [C]      (product nodes)
        #      \   /
        #       \ /
        #       [D]         (sum nodes)
        #       / \
        #      /   \
        #    [E]   [F]      (product nodes)
        #     |\   /|
        #     | \ / |
        #     |  X  |
        #     | / \ |
        #     |/   \|
        #    [G]   [H]      (leaf nodes)

        H = TorchGaussian([0], 0.0, 1.0)
        G = TorchGaussian([1], 0.0, 1.0)

        F = TorchProductNode([G, H], [0])
        E = TorchProductNode([G, H], [0])

        D = TorchSumNode([E, F], [0])

        C = TorchProductNode([D], [0])
        B = TorchProductNode([D], [0])

        A = TorchSumNode([B, C], [0])

        data = torch.randn(3, 2)

        hist = []

        def tracefunc(frame, event, arg, hist=hist):
            if event == "call" and frame.f_code.co_name in ["memoized_f", "log_likelihood"]:
                hist.append(frame.f_code.co_name)
            return tracefunc

        sys.settrace(tracefunc)

        log_likelihood(A, data)

        sys.settrace(None)

        self.assertTrue(
            hist
            == [
                "memoized_f",
                "log_likelihood",  # node A not cached: A
                "memoized_f",
                "log_likelihood",  # node B not cached: A -> B
                "memoized_f",
                "log_likelihood",  # node D not cached: A -> B -> D
                "memoized_f",
                "log_likelihood",  # node E not cached: A -> B -> D -> E
                "memoized_f",
                "log_likelihood",  # node G not cached: A -> B -> D -> E -> G
                "memoized_f",
                "log_likelihood",  # node H not cached: A -> B -> D -> E -> H
                "memoized_f",
                "log_likelihood",  # node F not cached: A -> B -> D -> F
                "memoized_f",  # node G cached:     A -> B -> D -> F -> G
                "memoized_f",  # node H cached:     A -> B -> D -> F -> H
                "memoized_f",
                "log_likelihood",  # node C not cached: A -> C
                "memoized_f",  # node D cached:     A -> C -> D
            ]
        )

    def test_node_memoization_ll(self):
        leaf0 = Gaussian(scope=[0], mean=0, stdev=1.0)
        leaf1 = Gaussian(scope=[0], mean=0, stdev=1.0)
        leaf2 = Gaussian(scope=[1], mean=0, stdev=1.0)
        leaf3 = Gaussian(scope=[1], mean=0, stdev=1.0)
        leaf4 = Gaussian(scope=[0], mean=0, stdev=1.0)
        leaf5 = Gaussian(scope=[0], mean=0, stdev=1.0)
        leaf6 = Gaussian(scope=[1], mean=0, stdev=1.0)
        leaf7 = Gaussian(scope=[1], mean=0, stdev=1.0)
        prod0 = IProductNode(children=[leaf0, leaf2], scope=[0, 1])
        prod1 = IProductNode(children=[leaf0, leaf3], scope=[0, 1])
        prod2 = IProductNode(children=[leaf1, leaf2], scope=[0, 1])
        prod3 = IProductNode(children=[leaf1, leaf3], scope=[0, 1])
        prod4 = IProductNode(children=[leaf4, leaf6], scope=[0, 1])
        prod5 = IProductNode(children=[leaf4, leaf7], scope=[0, 1])
        prod6 = IProductNode(children=[leaf5, leaf6], scope=[0, 1])
        prod7 = IProductNode(children=[leaf4, leaf7], scope=[0, 1])
        sum0 = ISumNode(
            children=[prod0, prod1, prod2, prod3, prod4, prod5, prod6, prod7],
            scope=[0, 1],
            weights=None,
        )
        sum1 = ISumNode(
            children=[prod0, prod1, prod2, prod3, prod4, prod5, prod6, prod7],
            scope=[0, 1],
            weights=None,
        )
        sum2 = ISumNode(
            children=[prod0, prod1, prod2, prod3, prod4, prod5, prod6, prod7],
            scope=[0, 1],
            weights=None,
        )
        spn = ISumNode(children=[sum0, sum1, sum2], scope=[0, 1], weights=None)

        hist = []

        def tracefunc(frame, event, arg, hist=hist):
            if event == "call" and frame.f_code.co_name in ["memoized_f", "likelihood"]:
                hist.append(frame.f_code.co_name)
            return tracefunc

        sys.settrace(tracefunc)

        # compute outputs for internal node graph
        likelihood(spn, np.random.randn(3, 7).reshape(-1, 7), SPN())

        sys.settrace(None)

        self.assertTrue(
            len(list(filter(lambda a: a != "memoized_f", hist))) == sum(list(_get_node_counts(spn)))
        )

    def test_memoization_rat_spn_ll(self):

        random_variables = set(range(0, 7))
        depth = 3
        replicas = 3
        region_graph = random_region_graph(random_variables, depth, replicas)
        context = RandomVariableContext(
            parametric_types=[Gaussian] * len(region_graph.root_region.random_variables)
        )

        num_nodes_root = 3
        num_nodes_region = 3
        num_nodes_leaf = 3
        rat_spn = RatSpn(region_graph, num_nodes_root, num_nodes_region, num_nodes_leaf, context)

        hist = []

        def tracefunc(frame, event, arg, hist=hist):
            if event == "call" and frame.f_code.co_name in ["memoized_f", "likelihood"]:
                hist.append(frame.f_code.co_name)
            return tracefunc

        sys.settrace(tracefunc)

        # compute outputs for rat spn module
        likelihood(rat_spn, np.random.randn(3, 7).reshape(-1, 7))

        sys.settrace(None)

        self.assertTrue(
            len(list(filter(lambda a: a != "memoized_f", hist)))
            == sum(list(_get_node_counts(rat_spn.output_nodes[0]))) + 1
        )


if __name__ == "__main__":
    unittest.main()
