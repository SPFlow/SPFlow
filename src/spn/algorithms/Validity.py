'''
Created on March 20, 2018

@author: Alejandro Molina
'''
from spn.structure.Base import Sum, Leaf, Product


def is_consistent(node):
    '''
    all children of a product node have different scope
    '''

    assert node is not None

    if len(node.scope) == 0:
        return False

    if isinstance(node, Leaf):
        return True

    if isinstance(node, Product):
        nscope = set(node.scope)

        allchildscope = set()
        sum_features = 0
        for child in node.children:
            sum_features += len(child.scope)
            allchildscope = allchildscope | set(child.scope)

        if allchildscope != set(nscope) or sum_features != len(allchildscope):
            return False


    return all(map(is_consistent, node.children))


def is_complete(node):
    '''
    all children of a sum node have same scope as the parent
    '''

    assert node is not None

    if len(node.scope) == 0:
        return False

    if isinstance(node, Leaf):
        return True

    if isinstance(node, Sum):
        nscope = set(node.scope)

        for child in node.children:
            if nscope != set(child.scope):
                return False

    return all(map(is_complete, node.children))

def is_aligned(node):
    if isinstance(node, Leaf):
        return True

    if isinstance(node, Sum) and len(node.children) != len(node.weights):
        return False

    return all(map(is_aligned, node.children))

def is_valid(node):
    return is_consistent(node) and is_complete(node) and is_aligned(node)
