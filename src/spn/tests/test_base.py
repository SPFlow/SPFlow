import unittest

from spn.structure.Base import Leaf, bfs


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


if __name__ == '__main__':
    unittest.main()
