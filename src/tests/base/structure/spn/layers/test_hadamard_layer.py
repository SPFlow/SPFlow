import unittest

import numpy as np

from spflow.base.structure.spn import HadamardLayer, marginalize
from spflow.meta.data import Scope

from ...general.nodes.dummy_node import DummyNode


class TestLayer(unittest.TestCase):
    def test_hadamard_layer_initialization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
            [
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
            ],
        ]

        # ----- check attributes after correct initialization -----

        l = HadamardLayer(child_partitions=input_partitions)
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([0, 1, 2, 3, 4]) for _ in range(len(l.nodes))]))
        # make sure order of nodes is correct (important)
        for indices, node in zip([[0, 1, 4, 5], [0, 2, 4, 6], [0, 3, 4, 7]], l.nodes):
            self.assertTrue(node.chs[0].input_ids == indices)

        # only one partition
        l = HadamardLayer(
            child_partitions=[
                [
                    DummyNode(Scope([1, 3])),
                    DummyNode(Scope([1, 3])),
                    DummyNode(Scope([1, 3])),
                ]
            ]
        )
        # make sure number of creates nodes is correct
        self.assertEqual(len(l.nodes), 3)
        # make sure scopes are correct
        self.assertTrue(np.all(l.scopes_out == [Scope([1, 3]) for _ in range(len(l.nodes))]))
        # make sure order of nodes is correct (important)
        for indices, node in zip([[0], [1], [2]], l.nodes):
            self.assertTrue(node.chs[0].input_ids == indices)

        # ----- no child partitions -----
        self.assertRaises(ValueError, HadamardLayer, [])

        # ----- empty partition -----
        self.assertRaises(ValueError, HadamardLayer, [[]])

        # ----- scopes inside partition differ -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([1])), DummyNode(Scope([2]))],
            ],
        )

        # ----- partitions of non-pair-wise disjoint scopes -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0]))],
                [DummyNode(Scope([0])), DummyNode(Scope([0]))],
            ],
        )

        # ----- invalid total outputs of partitions -----
        self.assertRaises(
            ValueError,
            HadamardLayer,
            [
                [DummyNode(Scope([0])), DummyNode(Scope([0]))],
                [
                    DummyNode(Scope([1])),
                    DummyNode(Scope([1])),
                    DummyNode(Scope([1])),
                ],
            ],
        )

    def test_hadamard_layer_structural_marginalization(self):

        # dummy partitios over pair-wise disjont scopes
        input_partitions = [
            [DummyNode(Scope([0]))],
            [
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
                DummyNode(Scope([1, 3])),
            ],
            [DummyNode(Scope([2]))],
            [
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
                DummyNode(Scope([4])),
            ],
        ]

        l = HadamardLayer(child_partitions=input_partitions)
        # should marginalize entire module
        l_marg = marginalize(l, [0, 1, 2, 3, 4])
        self.assertTrue(l_marg is None)
        # should marginalize entire partition
        l_marg = marginalize(l, [2])
        self.assertTrue(l_marg.scope == Scope([0, 1, 3, 4]))
        # should partially marginalize one partition
        l_marg = marginalize(l, [3])
        self.assertTrue(l_marg.scope == Scope([0, 1, 2, 4]))

        # ----- marginalize over non-scope rvs -----
        l_marg = marginalize(l, [5])
        self.assertTrue(
            l_marg.scopes_out
            == [
                Scope([0, 1, 2, 3, 4]),
                Scope([0, 1, 2, 3, 4]),
                Scope([0, 1, 2, 3, 4]),
            ]
        )

        # ----- pruning -----
        l = HadamardLayer(child_partitions=input_partitions[:2])

        l_marg = marginalize(l, [1, 3], prune=True)
        self.assertTrue(isinstance(l_marg, DummyNode))


if __name__ == "__main__":
    unittest.main()
