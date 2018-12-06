"""
Created on March 27, 2018

@author: Alejandro Molina
"""
from copy import deepcopy

from spn.algorithms.Validity import is_valid
from spn.structure.Base import Leaf, Sum, Product, assign_ids, get_nodes_by_type, get_parents, get_topological_order


def Compress(node):
    all_parents = get_parents(node)

    cache = {}

    for n in get_topological_order(node):

        params = (n.parameters, tuple(sorted(n.scope)))

        cached_node = cache.get(params, None)
        if cached_node is None:
            cache[params] = n
        else:
            for parent, pos in all_parents[n]:
                parent.children[pos] = cached_node

    assign_ids(node)
    val, msg = is_valid(node)
    assert val, msg
    return node


def Prune(node):
    v, err = is_valid(node)
    assert v, err
    nodes = get_nodes_by_type(node, (Product, Sum))

    while len(nodes) > 0:
        n = nodes.pop()

        n_type = type(n)
        is_sum = n_type == Sum

        i = 0
        while i < len(n.children):
            c = n.children[i]

            # if my children has only one node, we can get rid of it and link directly to that grandchildren
            if not isinstance(c, Leaf) and len(c.children) == 1:
                n.children[i] = c.children[0]
                continue

            if n_type == type(c):
                del n.children[i]
                n.children.extend(c.children)

                if is_sum:
                    w = n.weights[i]
                    del n.weights[i]

                    n.weights.extend([cw * w for cw in c.weights])
                continue

            i += 1
        if is_sum and i > 0:
            n.weights[0] = 1.0 - sum(n.weights[1:])

    if isinstance(node, (Product, Sum)) and len(node.children) == 1:
        node = node.children[0]

    assign_ids(node)
    v, err = is_valid(node)
    assert v, err
    return node


def Copy(node, validate=False):
    if validate:
        v, err = is_valid(node)
        assert v, err
    return deepcopy(node)


def SPN_Reshape(node, max_children=2):
    v, err = is_valid(node)
    assert v, err
    nodes = get_nodes_by_type(node, (Product, Sum))

    while len(nodes) > 0:
        n = nodes.pop()

        if len(n.children) <= max_children:
            continue

        # node has more than 2 nodes, create binary hierarchy
        new_children = []
        new_weights = []
        for i in range(0, len(n.children), max_children):
            children = n.children[i : i + max_children]

            if len(children) > 1:
                if isinstance(n, Product):
                    newChild = Product()
                    for c in children:
                        newChild.scope.extend(c.scope)
                    newChild.children.extend(children)
                    new_children.append(newChild)
                else:  # Sum
                    weights = n.weights[i : i + max_children]
                    branch_weight = sum(weights)
                    new_weights.append(branch_weight)

                    newChild = Sum()
                    newChild.scope.extend(children[0].scope)
                    newChild.children.extend(children)
                    newChild.weights.extend([w / branch_weight for w in weights])
                    newChild.weights[0] = 1.0 - sum(newChild.weights[1:])
                    new_children.append(newChild)
            else:
                new_children.extend(children)

                if isinstance(n, Sum):
                    new_weights.append(1.0 - sum(new_weights))

        n.children = new_children
        if isinstance(n, Sum):
            n.weights = new_weights
        nodes.append(n)

    assign_ids(node)
    v, err = is_valid(node)
    assert v, err
    return node
