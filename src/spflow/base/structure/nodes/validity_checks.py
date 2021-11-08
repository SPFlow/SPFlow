from typing import List
from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.module import Module
from spflow.base.structure.nodes import ILeafNode, INode, IProductNode, ISumNode
from spflow.base.structure.rat import RatSpn
import numpy as np


@dispatch(list)  # type: ignore[no-redef]
def _isvalid_spn(root_nodes: List[INode]) -> None:
    """Assert that there are no None-states in the SPN, ISumNodes are smooth,
       IProductNodes are decomposable and ILeafNodes don't have children

    Args:
        root_nodes:
            A list of INodes that are the roots/outputs of the (perhaps multi-class) SPN.

    """
    # assert all nodes via BFS. This section is not runtime-optimized yet
    nodes: List[INode] = list(root_nodes)

    while nodes:
        node: INode = nodes.pop(0)
        assert node.scope is not None
        assert None not in node.scope

        # assert that ISumNodes are smooth and weights sum up to 1
        if type(node) is ISumNode:
            assert node.children is not None
            assert None not in node.children
            assert node.weights is not None
            assert None not in node.weights
            assert node.weights.shape == node.weights.shape
            assert np.array(node.children).shape == node.weights.shape
            assert np.isclose(sum(node.weights), 1.0)
            for child in node.children:
                assert child.scope == node.scope
        # assert that IProductNodes are decomposable
        elif type(node) is IProductNode:
            assert node.children is not None
            assert None not in node.children
            assert node.scope == sorted([scope for child in node.children for scope in child.scope])
            length = len(node.children)
            # assert that each child's scope is true subset of IProductNode's scope (set<set = subset)
            for i in range(0, length):
                assert set(node.children[i].scope) < set(node.scope)
                # assert that all children's scopes are pairwise distinct (set&set = intersection)
                for j in range(i + 1, length):
                    assert not set(node.children[i].scope) & set(node.children[j].scope)
        # assert that ILeafNodes are actually leaves
        elif isinstance(node, ILeafNode):
            assert len(node.children) == 0
        else:
            raise ValueError("Node must be ISumNode, IProductNode, or a subclass of ILeafNode")

        if node.children:
            nodes.extend(list(set(node.children) - set(nodes)))


@dispatch(INode)  # type: ignore[no-redef]
def _isvalid_spn(root_node: INode) -> None:
    """Wrapper for SPNs with one root"""
    _isvalid_spn([root_node])


# multiple output nodes?
@dispatch(Module)  # type: ignore[no-redef]
def _isvalid_spn(module: Module) -> None:
    """Wrapper for Modules"""
    _isvalid_spn(module.output_nodes[0])
