'''
Created on March 27, 2018

@author: Alejandro Molina
'''
from spn.structure.Base import Leaf, Sum


def prune(node):
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
    newNode.children.extend(map(prune, node.children))
    if isinstance(newNode, Sum):
        newNode.weights.extend(node.weights)

    return newNode
