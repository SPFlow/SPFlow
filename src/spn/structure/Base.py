'''
Created on March 20, 2018

@author: Alejandro Molina
'''


class Node:
    def __init__(self):
        self.id = 0
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



def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (Sum, Product))])


def get_number_of_layers(node):
    if isinstance(node, Leaf):
        return 1

    return max(map(get_number_of_layers, node.children)) + 1

def rebuild_scopes_bottom_up(node):
    # this function is not safe (updates in place)
    if isinstance(node, Leaf):
        return node.scope

    new_scope = set()
    for c in node.children:
        new_scope.update(rebuild_scopes_bottom_up(c))
    node.scope = list(new_scope)
    return node.scope

def bfs(root, func):
    import collections

    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for node in node.children:
                if node not in seen:
                    seen.add(node)
                    queue.append(node)

def get_nodes_by_type(node, ntype=Node):

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result

def assign_ids(node, ids={}):

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)