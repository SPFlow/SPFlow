'''
Created on March 20, 2018

@author: Alejandro Molina
'''
import numpy as np
import collections


class Node(object):
    def __init__(self):
        self.id = 0
        self.scope = []

    @property
    def name(self):
        return "%sNode_%s" % (self.__class__.__name__, self.id)

    def __repr__(self):
        return self.name

    def __rmul__(self, weight):
        assert type(weight) == int or type(weight) == float
        self._tmp_weight = weight
        return self

    def __mul__(self, node):
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        result = Product()
        result.children.append(self)
        result.children.append(node)
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        assign_ids(result)
        return result

    def __add__(self, node):
        assert isinstance(node, Node)
        assert hasattr(node, "_tmp_weight"), "right node has no weight"
        assert hasattr(self, "_tmp_weight"), "left node has no weight"
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert set(node.scope) == (set(self.scope)), "children's scope are not the same"

        from numpy import isclose
        assert isclose(1.0, self._tmp_weight + node._tmp_weight), \
            "unnormalized weights, maybe trying to add many nodes at the same time?"

        result = Sum()
        result.children.append(self)
        result.weights.append(self._tmp_weight)
        result.children.append(node)
        result.weights.append(node._tmp_weight)
        result.scope.extend(self.scope)
        result._tmp_weight = self._tmp_weight + node._tmp_weight
        assign_ids(result)
        return result


class Sum(Node):
    def __init__(self, weights=None, children=None):
        Node.__init__(self)
        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children


class Product(Node):
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children


class Leaf(Node):
    def __init__(self, scope=None):
        Node.__init__(self)
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))


class Context:
    def __init__(self, meta_types=None, domains=None, parametric_types=None):
        self.meta_types = meta_types
        self.domains = domains
        self.parametric_types = parametric_types

        if self.meta_types is None and parametric_types is not None:
            self.meta_types = []
            for p in parametric_types:
                self.meta_types.append(p.type.meta_type)

    def get_meta_types_by_scope(self, scopes):
        return [self.meta_types[s] for s in scopes]

    def get_domains_by_scope(self, scopes):
        return [self.domains[s] for s in scopes]

    def add_domains(self, data):
        assert len(data.shape) == 2, "data is not 2D?"
        assert data.shape[1] == len(self.meta_types), "Data columns and metatype size doesn't match"

        from spn.structure.StatisticalTypes import MetaType
        domain = []

        for col in range(data.shape[1]):
            feature_meta_type = self.meta_types[col]
            domain_values = [np.min(data[:, col]), np.max(data[:, col])]

            if feature_meta_type == MetaType.REAL or feature_meta_type == MetaType.BINARY:
                domain.append(domain_values)
            elif feature_meta_type == MetaType.DISCRETE:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
            else:
                raise Exception("Unkown MetaType " + str(feature_meta_type))

        self.domains = np.asanyarray(domain)

        return self


def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (Sum, Product))])


def get_number_of_layers(node):
    node_depth = {}

    def count_layers(node):
        ndepth = node_depth.setdefault(node, 1)

        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, ndepth + 1)

    bfs(node, count_layers)

    return max(node_depth.values())


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
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)


def get_nodes_by_type(node, ntype=Node):
    assert node is not None

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result


def assign_ids(node, ids=None):
    if ids is None:
        ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)
    return node


def eval_spn_bottom_up(node, eval_functions, all_results=None, debug=False, **args):
    """
    Evaluates the spn bottom up


    :param node: spn root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, args**) for leave nodes and (node, [children results], args**)
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node. It is used to store intermediate results so that non-tree graphs can be computed in O(n) size of the network
    :param debug: whether to present progress information on the evaluation
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """
    # evaluating in reverse order, means that we compute all the children first then their parents
    nodes = reversed(get_nodes_by_type(node))

    if debug:
        from tqdm import tqdm
        nodes = tqdm(list(nodes))

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if not hasattr(node_type, '_eval_func'):
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)

    tmp_children_list = []
    len_tmp_children_list = 0
    for n in nodes:

        func = None

        try:
            func = n.__class__._eval_func[-1]
        except:
            pass

        if func is None:
            raise Exception("No lambda function associated with type: %s" % (n.__class__.__name__))

        if n.__class__._is_leaf:
            result = func(n, **args)
        else:
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                tmp_children_list[i] = all_results[n.children[i]]
            result = func(n, tmp_children_list[0:len_children], **args)

        all_results[n] = result

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]

    return all_results[node]


def eval_spn_top_down(root, eval_functions, all_results=None, parent_result=None, **args):
    """
    evaluates an spn top to down


    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, parent_results, args**) and returns [node, intermediate_result]. This intermediate_result will be passed to node as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    queue = collections.deque([(root, parent_result)])
    while queue:
        node, parent_result = queue.popleft()
        result = eval_functions[type(node)](node, parent_result, **args)
        all_results[node] = result
        if result is not None and not isinstance(node, Leaf):
            assert len(result) == len(node.children), "invalid function result for node %s" % (node.id)
            for i, node in enumerate(node.children):
                queue.append((node, result[i]))

    return all_results[root]
