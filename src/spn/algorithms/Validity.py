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
        # print(node.scope, '0 scope const')
        return False, "node %s has no scope" % (node.id)

    if isinstance(node, Leaf):
        return True, None

    if isinstance(node, Product):
        nscope = set(node.scope)

        allchildscope = set()
        sum_features = 0
        for child in node.children:
            sum_features += len(child.scope)
            # print('cs ', sum_features, child.scope, child.__class__.__name__, node.scope)
            allchildscope = allchildscope | set(child.scope)

        if allchildscope != set(nscope) or sum_features != len(allchildscope):
            # print(allchildscope, set(nscope), sum_features, len(allchildscope), 'cons')
            return False, "children of (prod) node %s don' have exclusive scope" % (node.id)

    for c in node.children:
        consistent, err = is_consistent(c)
        if not consistent:
            return consistent, err

    return True, None


def is_complete(node):
    '''
    all children of a sum node have same scope as the parent
    '''

    assert node is not None

    if len(node.scope) == 0:
        # print(node.scope, '0 scope')
        return False, "node %s has no scope" % (node.id)

    if isinstance(node, Leaf):
        return True, None

    if isinstance(node, Sum):
        nscope = set(node.scope)

        for child in node.children:
            if nscope != set(child.scope):
                # print(node.scope, child.scope, 'mismatch scope')
                return False, "children of (sum) node %s don't have the same scope as parent" % (node.id)

    for c in node.children:
        complete, err = is_complete(c)
        if not complete:
            return complete, err

    return True, None


def is_valid(node):
    a, err = is_consistent(node)
    if not a:
        return a, err

    b, err = is_complete(node)
    if not b:
        return b, err

    return True, None
