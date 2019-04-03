"""
Created on March 20, 2018

@author: Alejandro Molina
"""
from spn.structure.Base import Sum, Product, get_nodes_by_type, Max
import logging

logger = logging.getLogger(__name__)


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

#added
def is_max_complete(node):
    '''
    all children of a max node have same scope as the parent
    '''

    assert node is not None

    for max_node in reversed(get_nodes_by_type(node, Max)):
        nscope = set(max_node.scope)

        # if len(max_node.children) == 0:
        #     return False, "Max node %s has no children" % (max_node.id)

        for child in max_node.children:
            if nscope != set(child.scope):
                return False, "children of (max) node %s do not have the same scope as parent" % (max_node.id)

    return True, None


#added
def is_max_unique(node):
    '''
    each max node that corresponds to a decision variable D appears at most once in every path from root to leaves
    '''

    assert node is not None

    for max_node in reversed(get_nodes_by_type(node, Max)):
        check_children_max(max_node, max_node.name)
       
    return True, None
#added
def check_children_max(node, max_node_name):

    if isinstance(node, Max):
        if node.name == max_node_name:
            return False, "there are more than one same (max) node %s  in the path" % (node.id)
    for child in node.children:
        check_children_max(node, max_node_name)



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

#added
def is_valid_spmn(node, check_ids=True):

    if check_ids:
        val, err = has_valid_ids(node)
        if not val:
            return val, err

    for n in get_nodes_by_type(node):
        if len(n.scope) == 0:
            return False, "node %s has no scope" % (n.id)
        is_sum = isinstance(n, Sum)
        is_prod = isinstance(n, Product)
        is_max = isinstance(n, Max)
        if is_sum:
            if len(n.children) != len(n.weights):
                return False, "node %s has different children/weights" % (n.id)

        if is_sum or is_prod:
            if len(n.children) == 0:
                return False, "node %s has no children" % (n.id)
            
        if is_max:
            if len(n.children) > len(n.dec_values):
                return False, "node %s has different children/dec_vals" % (n.id)

    a, err = is_consistent(node)
    if not a:
        return a, err

    b, err = is_complete(node)
    if not b:
        return b, err

    c, err = is_max_complete(node)
    if not c:
        return c, err

    d, err = is_max_unique(node)
    if not d:
        return d, err
    
    return True, 'SPMN is valid'

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
