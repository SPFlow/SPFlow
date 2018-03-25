'''
Created on March 20, 2018

@author: Alejandro Molina
'''


class Node:
    def __init__(self):
        self.scope = []


class Sum(Node):
    def __init__(self):
        Node.__init__(self)
        self.weights = []
        self.children = []


class Product(Node):
    def __init__(self):
        Node.__init__(self)
        self.children = []


class Leaf(Node):
    def __init__(self):
        Node.__init__(self)


class Context:
    pass


def get_node_by_type(node, ntype=Node):
    result = []
    if isinstance(node, ntype):
        result.append(node)

    if not isinstance(node, Leaf):
        for c in node.children:
            result.extend(get_node_by_type(c, ntype))

    return result


def get_number_of_edges(node):
    return sum([len(c.children) for c in get_node_by_type(node, (Sum, Product))])


def get_number_of_layers(node):
    if isinstance(node, Leaf):
        return 1

    return max(map(get_number_of_layers, node.children)) + 1
