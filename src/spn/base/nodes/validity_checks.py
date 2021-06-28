from typing import List
from multimethod import multimethod
from spn.base.nodes.node import LeafNode, Node, ProductNode, SumNode

from spn.base.rat.rat_spn import RatSpn
import numpy as np


@multimethod
def _isvalid_spn(root_nodes: List[Node]) -> None:
    """Assert that there are no None-states in the SPN, SumNodes are smooth,
       ProductNodes are decomposable and LeafNodes don't have children

    Args:
        root_nodes:
            A list of Nodes that are the roots/outputs of the (perhaps multi-class) SPN.

    """
    # assert all nodes via BFS. This section is not runtime-optimized yet
    nodes: List[Node] = list(root_nodes)

    while nodes:
        node: Node = nodes.pop(0)
        assert node.scope is not None
        assert None not in node.scope

        # assert that SumNodes are smooth and weights sum up to 1
        if type(node) is SumNode:
            assert node.children is not None
            assert None not in node.children
            assert node.weights is not None
            assert None not in node.weights
            assert node.weights.shape == node.weights.shape
            assert np.array(node.children).shape == node.weights.shape
            assert np.isclose(sum(node.weights), 1.0)
            for child in node.children:
                assert child.scope == node.scope
        # assert that ProductNodes are decomposable
        elif type(node) is ProductNode:
            assert node.children is not None
            assert None not in node.children
            assert node.scope == sorted([scope for child in node.children for scope in child.scope])
            length = len(node.children)
            # assert that each child's scope is true subset of ProductNode's scope (set<set = subset)
            for i in range(0, length):
                assert set(node.children[i].scope) < set(node.scope)
                # assert that all children's scopes are pairwise distinct (set&set = intersection)
                for j in range(i + 1, length):
                    assert not set(node.children[i].scope) & set(node.children[j].scope)
        # assert that LeafNodes are actually leaves
        elif isinstance(node, LeafNode):
            assert len(node.children) == 0
        else:
            raise ValueError("Node must be SumNode, ProductNode, or a subclass of LeafNode")

        if node.children:
            nodes.extend(list(set(node.children) - set(nodes)))


@multimethod  # type: ignore[no-redef]
def _isvalid_spn(root_node: Node) -> None:
    """Wrapper for SPNs with one root"""
    _isvalid_spn([root_node])


@multimethod  # type: ignore[no-redef]
def _isvalid_spn(rat_spn: RatSpn) -> None:
    """Wrapper for RAT-SPNs"""
    _isvalid_spn(rat_spn.root_node)
