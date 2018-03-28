'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from spn.algorithms.Pruning import prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Sum, Leaf, rebuild_scopes_bottom_up, assign_ids


def marginalize(node, scope):
    assert isinstance(scope, set), "scope must be a set"


    def marg_recursive(node):
        node_scope = set(node.scope)

        if node_scope.issubset(scope):
            return None

        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception('Leaf Node with |scope| > 1')

            return node

        newNode = node.__class__()

        #a sum node gets copied with all its children, or gets removed completely
        if isinstance(node, Sum):
            newNode.weights.extend(node.weights)

        for i, c in enumerate(node.children):
            newChildren = marg_recursive(c)
            if newChildren is None:
                continue

            newNode.children.append(newChildren)
        return newNode

    newNode = marg_recursive(node)
    rebuild_scopes_bottom_up(newNode)
    newNode = prune(newNode)
    assert is_valid(newNode)
    assign_ids(node)
    return newNode




