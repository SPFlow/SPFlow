'''
Created on March 27, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Leaf, Sum, Product, assign_ids, get_nodes_by_type


def Prune(node):
    if isinstance(node, Leaf):
        return node

    while True:

        pruneNeeded = any(map(lambda c: isinstance(c, type(node)), node.children))

        if not pruneNeeded:
            break

        newNode = node.__class__()
        newNode.scope.extend(node.scope)

        newChildren = []
        newWeights = []
        for i, c in enumerate(node.children):
            if type(c) != type(newNode):
                newChildren.append(c)
                if isinstance(newNode, Sum):
                    newWeights.append(node.weights[i])
                continue
            else:
                for j, gc in enumerate(c.children):
                    newChildren.append(gc)
                    if isinstance(newNode, Sum):
                        newWeights.append(node.weights[i] * c.weights[j])

        newNode.children.extend(newChildren)

        if isinstance(newNode, Sum):
            newNode.weights.extend(newWeights)

        node = newNode

    while not isinstance(node, Leaf) and len(node.children) == 1:
        node = node.children[0]

    if isinstance(node, Leaf):
        return node

    newNode = node.__class__()
    newNode.scope.extend(node.scope)
    newNode.children.extend(map(Prune, node.children))
    if isinstance(newNode, Sum):
        newNode.weights.extend(node.weights)

    return newNode


def SPN_Reshape(node, max_children=2):
    nodes = get_nodes_by_type(node, (Product, Sum))

    while len(nodes) > 0:
        n = nodes.pop()

        if len(n.children) <= max_children:
            continue

        # node has more than 2 nodes, create binary hierarchy
        new_children = []
        for children in [n.children[i:i + max_children] for i in range(0, len(n.children), max_children)]:
            if len(children) == max_children:
                if isinstance(n, Product):
                    newChild = Product()
                else:
                    newChild = Sum()
                    newChild.weights
                newChild.children.extend(children)
                new_children.append(newChild)
            else:
                new_children.extend(children)

        n.children = new_children
        nodes.append(n)

    assign_ids(node)
    assert is_valid(node)
    return node
