"""
Created on May 31, 2021

@authors: Bennet Wittelsbach, Kevin Huy Nguyen

This file provides the structure and construction algorithm for abstract RAT-SPNs, which are
depending on the abstract the abstract nodes and the region graph.
"""
from region_graph import *
from Node import *


class RATSPN:
    """The RATSPN holds all sum and product nodes and relies on the structure of the region graph.

    Attributes:
        nodes:
            A list of root sum nodes.
    """

    def __init__(self, nodes: Optional[List["SumNode"]]) -> None:
        self.nodes = nodes if nodes else []


def construct_spn_from_region_graph(
    region_graph: RegionGraph, num_root_sum: int, num_int_sum: int, num_inp_distr: int
):
    """Constructs the RAT-SPN from the structure of a given region graph.

    This function traverses through the given region graph twice top-down. The first iteration it initializes all the
    sum and leaf nodes and equips them to their corresponding region to remember the structure. The second iteration
    initializes all the product nodes and connects them to their children and parents. The product nodes form
    cross-products of nodes, which are co-children of the partition. The product nodes then get connected to all the
    sum nodes in their parent region.

    Args:
        region_graph:
            The RegionGraph which given.
        num_root_sum:
            The number of sum nodes in the root region.
        num_int_sum:
            The number of sum nodes in the internal regions.
        num_inp_distr:
            The number of leaf nodes in the leaf regions
    """
    nodes: List[Any] = list(region_graph.root_region.partitions)

    # initialize RAT-SPN
    rat_spn = RATSPN(nodes=[])

    # populate RAT-SPN with root sum nodes
    i = 0
    while i < num_root_sum:
        root_node = SumNode(
            children=[], scope=list(region_graph.root_region.random_variables), weights=[]
        )
        rat_spn.nodes.append(root_node)
        i += 1

    # first traversion through region graph for creation of sum nodes
    while nodes:
        node: Any = nodes.pop(0)

        if type(node) is Partition:
            nodes.extend(node.regions)

        elif type(node) is Region:
            nodes.extend(node.partitions)
            # if the region has no partition as children, it is a leaf node
            if node.partitions == set():
                node.nodes = []
                i = 0
                while i < num_inp_distr:
                    leaf = LeafNode(children=None, scope=list(node.random_variables))
                    node.nodes.append(leaf)
                    i += 1
            # if the region has children it is an internal sum node
            else:
                node.nodes = []
                i = 0
                while i < num_int_sum:
                    sum = SumNode(children=[], scope=list(node.random_variables), weights=[])
                    node.nodes.append(sum)
                    i += 1
        else:
            raise ValueError("Node must be Region or Partition")

    # second traversion for product nodes
    # staggered_nodes collects the parent regions of the product nodes, so that we can connect them
    staggered_nodes = [rat_spn]
    nodes = list(region_graph.root_region.partitions)
    while nodes:
        node = nodes.pop(0)

        if type(node) is Partition:
            nodes.extend(node.regions)
            node.nodes = []
            # sub_region_nodes is a list of lists of all the nodes in the child regions of the current partition
            sub_region_nodes = [region.nodes for region in node.regions]
            # creates product nodes and connects them in a way that the nodes of the child regions form cross products.
            for sub_region_node_1 in sub_region_nodes[0]:
                for sub_region_node_2 in sub_region_nodes[1]:
                    scope = [sub_region_node_1.scope + sub_region_node_2.scope]
                    product_node = ProductNode(
                        children=[sub_region_node_1, sub_region_node_2], scope=scope
                    )
                    print("product")
                    print(scope)
                    node.nodes.append(product_node)
                    print("children")
                    print(sub_region_node_1)
                    print(sub_region_node_2)
                    print("parent")
                    # connect the product node to all the sum nodes of its parent region
                    for sum_node in staggered_nodes[0].nodes:
                        print(staggered_nodes[0])
                        sum_node.children.append(product_node)
            # after the product nodes of a partition is connected to the sum nodes of a parent region,
            # we can remove the parent region in staggered_nodes
            # exception is the root region as it can have multiple partitions, so we do not remove the parent region
            if type(staggered_nodes[0]) is not RATSPN:
                del staggered_nodes[0]
        elif type(node) is Region:
            nodes.extend(node.partitions)
        else:
            raise ValueError("Node must be Region or Partition")
        # if all objects in nodes are regions, it means that all parent regions in staggered nodes have been processed
        # we replace staggered_nodes with the parent regions of the next layer
        if all(isinstance(x, Region) for x in nodes):
            staggered_nodes = nodes.copy()
    return rat_spn


if __name__ == "__main__":
    region_graph = random_region_graph(X=set(range(1, 8)), depth=2, replicas=2)
    rat_spn = construct_spn_from_region_graph(region_graph, 3, 2, 2)
    print(rat_spn.nodes)
