"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach

This file provides the basic components to build abstract probabilistic circuits, like SumNode, ProductNode,
and LeafNode.
"""

from typing import List, Tuple, cast, Callable, Set, Type, Deque, Optional, Dict, Union
from multipledispatch import dispatch  # type: ignore
import numpy as np
import collections
from collections import deque, OrderedDict


class Node:
    """Base class for all types of nodes in an SPN

    Attributes:
        children:
            A list of Nodes containing the children of this Node, or None.
        scope:
            A list of integers containing the scopes of this Node, or None.
        value:
            A float representing the value of the node. nan-value represents a node, with its value not calculated yet.
    """

    def __init__(self, children: List["Node"], scope: List[int]) -> None:
        self.children = children
        self.scope = scope
        self.value: float = np.nan

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.scope}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return 1

    def print_treelike(self, prefix: str = "") -> None:
        """
        Ad-hoc method to print structure of node and children (for debugging purposes).
        """
        print(prefix + f"{self.__class__.__name__}: {self.scope}")

        for child in self.children:
            child.print_treelike(prefix=prefix + "    ")

    def equals(self, other: "Node") -> bool:
        """
        Checks whether two objects are identical by comparing their class, scope and children (recursively).
        """
        return (
            type(self) is type(other)
            and self.scope == other.scope
            and all(map(lambda x, y: x.equals(y), self.children, other.children))
        )


class ProductNode(Node):
    """A ProductNode provides a factorization of its children,
    i.e. ProductNodes in SPNs have children with distinct scopes"""

    def __init__(self, children: List[Node], scope: List[int]) -> None:
        super().__init__(children=children, scope=scope)


class SumNode(Node):
    """A SumNode provides a weighted mixture of its children, i.e. SumNodes in SPNs have children with identical scopes

    Attributes:
        weights:
            A np.array of floats assigning a weight value to each of the SumNode's children.

    """

    def __init__(self, children: List[Node], scope: List[int], weights: np.ndarray) -> None:
        super().__init__(children=children, scope=scope)
        self.weights = weights

    def equals(self, other: Node) -> bool:
        """
        Checks whether two objects are identical by comparing their class, scope, children (recursively) and weights.
        Note that weight comparison is done approximately due to numerical issues when conversion between graph
        representations.
        """
        if type(other) is SumNode:
            other = cast(SumNode, other)
            return (
                super().equals(other)
                and np.allclose(self.weights, other.weights, rtol=1.0e-5)
                and self.weights.shape == other.weights.shape
            )
        else:
            return False


class LeafNode(Node):
    """A LeafNode provides a probability distribution over the random variables in its scope"""

    def __init__(self, scope: List[int]) -> None:
        super().__init__(children=[], scope=scope)


@dispatch(list)  # type: ignore[no-redef]
def _print_node_graph(root_nodes: List[Node]) -> None:
    """Prints all unique nodes of a node graph in BFS fashion.

    Args:
        root_nodes:
            A list of Nodes that are the roots/outputs of the (perhaps multi-class) SPN.
    """
    nodes: List[Node] = list(root_nodes)
    while nodes:
        node: Node = nodes.pop(0)
        print(node)
        nodes.extend(list(set(node.children) - set(nodes)))


@dispatch(Node)  # type: ignore[no-redef]
def _print_node_graph(root_node: Node) -> None:
    """Wrapper for SPNs with single root node"""
    _print_node_graph([root_node])


@dispatch(list)  # type: ignore[no-redef]
def _get_node_counts(root_nodes: List[Node]) -> Tuple[int, int, int]:
    """Count the # of unique SumNodes, ProductNodes, LeafNodes in an SPN with arbitrarily many root nodes.

    Args:
        root_nodes:
            A list of Nodes that are the roots/outputs of the (perhaps multi-class) SPN.
    """
    nodes: List[Node] = root_nodes
    n_sumnodes = 0
    n_productnodes = 0
    n_leaves = 0

    while nodes:
        node: Node = nodes.pop(0)
        if type(node) is SumNode:
            n_sumnodes += 1
        elif type(node) is ProductNode:
            n_productnodes += 1
        elif isinstance(node, LeafNode):
            n_leaves += 1
        else:
            raise ValueError("Node must be SumNode, ProductNode, or LeafNode")
        nodes.extend(list(set(node.children) - set(nodes)))

    return n_sumnodes, n_productnodes, n_leaves


@dispatch(Node)  # type: ignore[no-redef]
def _get_node_counts(root_node: Node) -> Tuple[int, int, int]:
    """Wrapper for SPNs with single root node"""
    return _get_node_counts([root_node])


@dispatch(list)  # type: ignore[no-redef]
def _get_leaf_nodes(root_nodes: List[Node]) -> List[Node]:
    """Returns a list of leaf nodes to populate.

    Args:
        root_nodes:
            A list of Nodes that are the roots/outputs of the (perhaps multi-class) SPN.
    """
    nodes: List[Node] = root_nodes
    leaves: List[Node] = []
    id_counter = 0
    while nodes:
        node: Node = nodes.pop(0)
        if type(node) is LeafNode:
            leaves.append(node)
        nodes.extend(list(set(node.children) - set(nodes)))
        id_counter += 1
    return leaves


@dispatch(Node)  # type: ignore[no-redef]
def _get_leaf_nodes(root_node: Node) -> List[Node]:
    """Wrapper for SPNs with single root node"""
    return _get_leaf_nodes([root_node])


########################################################################################################################
# multi für single vs multiple root nodes?
def bfs(root: Node, func: Callable):
    """Iterates through SPN in breadth first order and calls func on all nodes in SPN.

    Args:
        root:
            SPN root node.
        func:
            function to call on all the nodes.
    """

    seen: Set = {root}
    queue: Deque[Node] = collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, LeafNode):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)


def get_nodes_by_type(node: Node, ntype: Union[Type, Tuple[Type, ...]] = Node) -> List[Node]:
    """Iterates SPN in breadth first order and collects nodes of type ntype..

    Args:
        node:
            SPN root node.
        ntype:
            Type of nodes to get. If not specified it will look for any node.

    Returns: List of nodes in SPN which fit ntype.
    """
    assert node is not None
    result: List[Node] = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)
    return result


def get_topological_order(node: Node) -> List[Node]:
    """
    Evaluates the spn bottom up using functions specified for node types.

    Args:
        node:
            SPN root node.

    Returns: List of nodes in SPN in their order specified by their structure.
    """
    nodes: List[Node] = get_nodes_by_type(node)

    parents: "OrderedDict[Node, List]" = OrderedDict({node: []})
    in_degree: "OrderedDict[Node, int]" = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, LeafNode):
            for c in n.children:
                parent_list: Optional[List[Optional[Node]]] = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    S: Deque = deque()  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            S.appendleft(u)

    L: List[Node] = []  # Empty list that will contain the sorted elements

    while S:
        n = S.pop()  # remove a node n from S
        L.append(n)  # add n to tail of L

        for m in parents[n]:  # for each node m with an edge e from n to m do
            in_degree_m: int = in_degree[m] - 1  # remove edge e from the graph
            in_degree[m] = in_degree_m
            if in_degree_m == 0:  # if m has no other incoming edges then
                S.appendleft(m)  # insert m into S

    assert len(L) == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L
