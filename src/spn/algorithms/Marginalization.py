'''
Created on March 21, 2018

@author: Alejandro Molina
'''
from copy import deepcopy

from spn.algorithms.Pruning import prune
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Sum, Leaf, assign_ids


def marginalize(node, keep):
    #keep must be a set of features that you want to keep
    
    assert isinstance(keep, set), "scope must be a set"

    def marg_recursive(node):
        new_node_scope = keep.intersection(set(node.scope))

        if isinstance(node, Leaf):
            if len(node.scope) > 1:
                raise Exception('Leaf Node with |scope| > 1')

            if len(new_node_scope) == 0:
                # we are summing out this node
                return None

            return deepcopy(node)

        newNode = node.__class__()

        if isinstance(node, Sum):
            newNode.weights.extend(node.weights)

        for c in node.children:
            new_c = marg_recursive(c)
            if new_c is None:
                continue
            newNode.children.append(new_c)

        newNode.scope.extend(new_node_scope)
        return newNode


    newNode = marg_recursive(node)
    #newNode = prune(newNode)
    assert is_valid(newNode)
    assign_ids(newNode)
    return newNode
