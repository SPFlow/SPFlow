'''
Edges Base Class for the Layerwise computation
'''


class ProductEdges(object):

    def __init__(self, child, parent, connections):
        self.parent = parent
        self.child = child
        self.connections = connections

class SumEdges(object):

    def __init__(self, child, parent, connections, weights):
        self.parent = parent
        self.child = child
        self.connections = connections
        self.weights = weights
