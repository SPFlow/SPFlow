import unittest

from spn.structure.graph import Node


class TestNode(unittest.TestCase):
    def test_spn(self):
        # dieser Test ist noch ziemlich nutzlos, kann aber spaeter als Schema für gelernte SPNs und Tests prinzipiell dienen
        spn: Node.Node = Node.SumNode(
            [
                Node.ProductNode(
                    children=[Node.LeafNode(scope=[1]), Node.LeafNode(scope=[2])],
                    scope=[1, 2],
                ),
                Node.ProductNode(
                    children=[Node.LeafNode(scope=[1]), Node.LeafNode(scope=[2])],
                    scope=[1, 2],
                ),
            ],
            scope=[1, 2],
            weights=[0.3, 0.7],
        )

        # assert all nodes via BFS. This section is not runtime-optimized
        nodes: list[Node.Node] = [spn]
        while nodes:
            node: Node.Node = nodes.pop()
            self.assertIsNotNone(node.scope)

            # assert that SumNodes are smooth and weights sum up to 1
            if type(node) is Node.SumNode:
                self.assertIsNotNone(node.children)
                self.assertIsNotNone(node.weights)
                self.assertAlmostEqual(sum(node.weights), 1.0)
                for child in node.children:
                    self.assertEqual(child.scope, node.scope)

            # assert that ProductNodes are decomposable
            elif type(node) is Node.ProductNode:
                self.assertIsNotNone(node.children)
                self.assertEqual(
                    node.scope,
                    [scope for child in node.children for scope in child.scope],
                )
                length = len(node.children)
                # assert that each child's scope is true subset of ProductNode's scope (set<set = subset)
                for i in range(0, length):
                    self.assertTrue(set(node.children[i].scope) < set(node.scope))
                    # assert that all children's scopes are pairwise distinct (set&set = intersection)
                    for j in range(i + 1, length):
                        self.assertFalse(
                            set(node.children[i].scope) & set(node.children[j].scope)
                        )

            # assert that LeafNodes are actually leaves
            elif isinstance(node, Node.LeafNode):
                self.assertIsNone(node.children)
            else:
                self.AssertionError(
                    "Node must be SumNode, ProductNode, or a subclass of LeafNode"
                )

            if node.children:
                nodes.extend(node.children)


if __name__ == "__main__":
    unittest.main()
