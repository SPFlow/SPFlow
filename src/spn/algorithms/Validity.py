"""
Created on March 20, 2018

@author: Alejandro Molina
"""
from spn.structure.Base import Sum, Product, get_nodes_by_type


def is_consistent(node):
    """
    all children of a product node have different scope
    """

    assert node is not None

    allchildscope = set()
    for prod_node in reversed(get_nodes_by_type(node, Product)):
        nscope = set(prod_node.scope)

        if len(prod_node.children) == 0:
            return False, "Product node %s has no children" % (prod_node.id)

        allchildscope.clear()
        sum_features = 0
        for child in prod_node.children:
            sum_features += len(child.scope)
            allchildscope.update(child.scope)

        if allchildscope != nscope or sum_features != len(allchildscope):
            return (False, "children of (prod) node %s do not have exclusive scope" % (prod_node.id))

    return True, None


def is_complete(node):
    """
    all children of a sum node have same scope as the parent
    """

    assert node is not None

    for sum_node in reversed(get_nodes_by_type(node, Sum)):
        nscope = set(sum_node.scope)

        if len(sum_node.children) == 0:
            return False, "Sum node %s has no children" % (sum_node.id)

        for child in sum_node.children:
            if nscope != set(child.scope):
                return (False, "children of (sum) node %s do not have the same scope as parent" % (sum_node.id))

    return True, None


def is_valid(node, check_ids=True):

    if check_ids:
        val, err = has_valid_ids(node)
        if not val:
            return val, err

    for n in get_nodes_by_type(node):
        if len(n.scope) == 0:
            return False, "node %s has no scope" % (n.id)
        is_sum = isinstance(n, Sum)
        is_prod = isinstance(n, Product)

        if is_sum:
            if len(n.children) != len(n.weights):
                return False, "node %s has different children/weights" % (n.id)

        if is_sum or is_prod:
            if len(n.children) == 0:
                return False, "node %s has no children" % (n.id)

    a, err = is_consistent(node)
    if not a:
        return a, err

    b, err = is_complete(node)
    if not b:
        return b, err

    return True, None


def has_valid_ids(node):
    ids = set()
    all_nodes = get_nodes_by_type(node)
    for n in all_nodes:
        ids.add(n.id)

    if len(ids) != len(all_nodes):
        return False, "Nodes are missing ids or there are repeated ids"

    if min(ids) != 0:
        return False, "Node ids not starting at 0"

    if max(ids) != len(ids) - 1:
        return False, "Node ids not consecutive"

    return True, None
