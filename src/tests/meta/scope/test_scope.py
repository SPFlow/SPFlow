from spflow.meta.scope.scope import Scope
import numpy as np
import unittest


class TestScope(unittest.TestCase):
    def test_scope_initialization(self):

        # correct initialization
        Scope([0, 4, 3], [1, 5, 2])

        # duplicated rvs
        self.assertRaises(ValueError, Scope, [0, 0])
        self.assertRaises(ValueError, Scope, [], [0, 0])

        # query and evidence non-disjoint
        self.assertRaises(ValueError, Scope, [0], [0])

        # empty query, non-empty evidence
        self.assertRaises(ValueError, Scope, [], [0])

        # negative scope rvs
        self.assertRaises(ValueError, Scope, [-1], [])
        self.assertRaises(ValueError, Scope, [0], [-1])

    def test_scope_methods(self):

        # equality
        self.assertEqual(Scope([1, 2], [3, 4]), Scope([2, 1], [4, 3]))
        self.assertNotEqual(Scope([1, 2], [3, 4]), Scope([1, 3], [2, 4]))

        # length
        self.assertEqual(len(Scope([0, 1])), 2)
        self.assertEqual(len(Scope([0, 1])), len(Scope([0, 1], [2])))
        self.assertEqual(len(Scope([])), 0)

        # is empty
        self.assertTrue(Scope([]).isempty())
        self.assertFalse(Scope([0]).isempty())

        # equal query
        self.assertTrue(Scope([1, 0], [2]).equal_query(Scope([0, 1], [3])))
        self.assertFalse(Scope([1, 0], [3]).equal_query(Scope([0, 2], [3])))

        # equal evidence
        self.assertTrue(Scope([1, 0], [2]).equal_evidence(Scope([0, 1], [2])))
        self.assertFalse(Scope([1, 0], [2]).equal_evidence(Scope([0, 1], [3])))

        # is disjoint
        self.assertTrue(Scope([0, 1], [2, 3]).isdisjoint(Scope([2, 3], [0, 1])))
        self.assertFalse(Scope([0, 1], [2]).isdisjoint(Scope([1, 2], [0])))

        # union
        s = Scope([0, 1], [2, 3])
        self.assertRaises(ValueError, s.union, Scope([1, 2], [0, 3]))
        self.assertEqual(
            Scope([0, 1], [3]).union(Scope([1, 2], [4])),
            Scope([0, 1, 2], [3, 4]),
        )

        # all pairwise disjoint
        self.assertTrue(
            Scope.all_pairwise_disjoint([Scope([i]) for i in range(5)])
        )
        self.assertFalse(Scope.all_pairwise_disjoint([Scope([0]), Scope([0])]))

        # all equal
        self.assertTrue(Scope.all_equal([Scope([0]), Scope([0])]))
        self.assertFalse(Scope.all_equal([Scope([1]), Scope([0])]))


if __name__ == "__main__":
    unittest.main()
