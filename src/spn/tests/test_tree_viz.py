import unittest

from spn.structure.Base import Leaf, bfs

from spn.io.plot.TreeVisualization import get_newick

class TestBase(unittest.TestCase):

    def __init__(self):
        D = Leaf(scope=[0])
        E = Leaf(scope=[0])
        F = Leaf(scope=[0])

        B = 0.5 * D + 0.5 * E
        C = 0.5 * E + 0.5 * F

        A = 0.5 * B + 0.5 * C


    def test_simple_newick_string(self):

        newick_string = get_newick(A)

        self.assertEqual(newick_string, '((LeafNode_3:1,LeafNode_4:1)SumNode_1:1,(LeafNode_4:1,LeafNode_5:1)SumNode_2:1);')

        A = 0.5 * C + 0.5 * B

        

if __name__ == '__main__':
    unittest.main()
