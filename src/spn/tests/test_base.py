import unittest

from spn.structure.Base import Leaf, bfs, get_topological_order


class TestBase(unittest.TestCase):
    def test_bfs(self):
        D = Leaf(scope=[0])
        E = Leaf(scope=[0])
        F = Leaf(scope=[0])

        B = 0.5 * D + 0.5 * E
        C = 0.5 * E + 0.5 * F

        A = 0.5 * B + 0.5 * C

        result = []

        def add_node(node):
            result.append(node)

        bfs(A, add_node)

        self.assertEqual(result[0], A)
        self.assertEqual(result[1], B)
        self.assertEqual(result[2], C)
        self.assertEqual(result[3], D)
        self.assertEqual(result[4], E)
        self.assertEqual(result[5], F)

        self.assertEqual(len(result), 6)

    def test_topological_order_for_tree(self):
        D = Leaf(scope=[0])
        E = Leaf(scope=[0])
        F = Leaf(scope=[0])

        B = 0.5 * D + 0.5 * E
        C = 0.5 * E + 0.5 * F

        A = 0.5 * B + 0.5 * C
        A.aname = "A"
        B.aname = "B"
        C.aname = "C"
        D.aname = "D"
        E.aname = "E"
        F.aname = "F"

        result = get_topological_order(A)

        self.assertEqual(result[0], D)
        self.assertEqual(result[1], E)
        self.assertEqual(result[2], F)
        self.assertEqual(result[3], B)
        self.assertEqual(result[4], C)
        self.assertEqual(result[5], A)
        self.assertEqual(len(result), 6)

    def test_topological_order_for_non_tree(self):
        D = Leaf(scope=[0])
        E = Leaf(scope=[0])
        F = Leaf(scope=[0])

        B = 0.5 * D + 0.5 * E
        C = 0.5 * E + 0.5 * F

        H = 0.5 * D + 0.5 * E
        I = 0.5 * D + 0.5 * E

        G = 0.5 * H + 0.5 * I
        A = 0.5 * B + 0.5 * C
        Z = 0.5 * A + 0.5 * G
        Z.aname = "Z"
        A.aname = "A"
        B.aname = "B"
        C.aname = "C"
        D.aname = "D"
        E.aname = "E"
        F.aname = "F"
        G.aname = "G"
        H.aname = "H"
        I.aname = "I"

        result = get_topological_order(Z)

        self.assertEqual(result[0], D)
        self.assertEqual(result[1], E)
        self.assertEqual(result[2], F)
        self.assertEqual(result[3], B)
        self.assertEqual(result[4], H)
        self.assertEqual(result[5], I)
        self.assertEqual(result[6], C)
        self.assertEqual(result[7], G)
        self.assertEqual(result[8], A)
        self.assertEqual(result[9], Z)
        self.assertEqual(len(result), 10)


if __name__ == "__main__":
    unittest.main()
