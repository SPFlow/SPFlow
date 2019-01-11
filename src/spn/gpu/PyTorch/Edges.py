"""
Edges Base Class for the Layerwise computation
"""


class ProductEdges(object):
    def __init__(self, child, parent, connections):
        """
        :param child: the child nodes/leaves
        :param parent: the parent nodes/leaves
        :param connections: the connections between the parent and the child (1 if
        the connection exists and 0 otherwise)
        """
        self.parent = parent
        self.child = child
        self.connections = connections


class SumEdges(object):
    def __init__(self, child, parent, connections, weights):
        """
        :param child: the child nodes/leaves
        :param parent: the parent nodes/leaves
        :param connections: the connections between the parent and the child (1 if
        the connection exists and 0 otherwise)
        :param weights: The weights on the sum edges
        """
        self.parent = parent
        self.child = child
        self.connections = connections
        self.weights = weights
