import unittest

from spn.structure.Base import Leaf, bfs

from spn.io.plot.TreeVisualization import get_newick


class TestBase(unittest.TestCase):
    def setUp(self):
        self.D = Leaf(scope=[0])
        self.E = Leaf(scope=[0])
        self.F = Leaf(scope=[0])

        self.B = 0.5 * self.D + 0.5 * self.E
        self.C = 0.5 * self.E + 0.5 * self.F

        self.A = 0.5 * self.B + 0.5 * self.C

    def test_simple_newick_string(self):

        newick_string = get_newick(self.A)

        self.assertEqual(newick_string, "((LeafNode_3__:1,LeafNode_4__:1)Σ:1,(LeafNode_4__:1,LeafNode_5__:1)Σ:1);")

        A = 0.5 * self.C + 0.5 * self.B


if __name__ == "__main__":
    unittest.main()
