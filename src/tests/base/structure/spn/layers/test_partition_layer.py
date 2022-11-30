import itertools
import unittest

import numpy as np

from spflow.base.structure.spn import PartitionLayer, marginalize
from spflow.meta.data import Scope

from ...general.nodes.dummy_node import DummyNode


class TestLayer(unittest.TestCase):
    def test_partition_layer_initialization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
        ]

        # ----- check attributes after correct initialization -----

        l = PartitionLayer(child_partitions=input_partitions)
        # make sure number of creates nodes is correct
        self.assertEqual(
            len(l.nodes),
            np.prod([len(partition) for partition in input_partitions]),
        )
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0, 1, 2, 3]) for _ in range(len(l.nodes))]))
        # make sure order of nodes is correct (important)
        for indices, node in zip(
            itertools.product(
                [0, 1],
                [2, 3, 4],
                [
                    5,
                ],
            ),
            l.nodes,
        ):
            self.assertTrue(node.children[0].input_ids == indices)

        # ----- no child partitions -----
        self.assertRaises(ValueError, PartitionLayer, [])

        # ----- empty partition -----
        self.assertRaises(ValueError, PartitionLayer, [[]])

        # ----- scopes inside partition differ -----
        self.assertRaises(
            ValueError,
            PartitionLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([1])), DummyNode(Scope([2]))],
            ],
        )

        # ----- partitions of non-pair-wise disjoint scopes -----
        self.assertRaises(
            ValueError,
            PartitionLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            ],
        )

    def test_partition_layer_structural_marginalization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
        ]

        l = PartitionLayer(child_partitions=input_partitions)
        # should marginalize entire module
        l_marg = marginalize(l, [0, 1, 2, 3])
        self.assertTrue(l_marg is None)
        # should marginalize entire partition
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scope == Scope([0, 1, 3]))
        # should partially marginalize one partition
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scope == Scope([0, 1, 2]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [4])
        self.assertTrue(
            l_marg.scopes_out
            == [
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
                Scope([0, 1, 2, 3]),
            ]
        )

        # ----- pruning -----
        l = PartitionLayer(child_partitions=input_partitions[1:])

        l_marg = marginalize(l, [1, 3], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))


if __name__ == "__main__":
    unittest.main()
