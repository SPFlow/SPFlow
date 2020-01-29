from scipy.sparse import csr_matrix, lil_matrix, dok_matrix

from spn.algorithms.TransformStructure import Copy, Prune

from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical

from spn.structure.Base import get_depth, Sum, Product, rebuild_scopes_bottom_up, assign_ids, \
    get_topological_order_layers
import numpy as np

from contextlib import contextmanager
from timeit import default_timer
from tqdm import tqdm


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def complete_layers(layer_nodes, current_node_type=Sum, depth=None):
    # all leaves should be at same depth
    root_layer = False
    if depth is None:
        root_layer = True
        depth = get_depth(layer_nodes[0])

    if depth == 2:
        return

    children_layer = []
    if current_node_type == Sum:
        for i in range(len(layer_nodes)):
            n = layer_nodes[i]
            assert isinstance(n, Sum)
            for j in range(len(n.children)):
                c = n.children[j]
                if not isinstance(c, Product):
                    n.children[j] = Product([c])
            children_layer.extend(n.children)
        children_layer_type = Product
    elif current_node_type == Product:
        for i in range(len(layer_nodes)):
            n = layer_nodes[i]
            assert isinstance(n, Product)
            for j in range(len(n.children)):
                c = n.children[j]
                if not isinstance(c, Sum):
                    n.children[j] = Sum([1.0], [c])
            children_layer.extend(n.children)
        children_layer_type = Sum
    else:
        raise Exception('node type' + str(current_node_type))

    complete_layers(children_layer, current_node_type=children_layer_type, depth=depth - 1)

    if root_layer:
        rebuild_scopes_bottom_up(layer_nodes[0])
        assign_ids(layer_nodes[0])


def get_scope(layer, children_layer, sparse):
    children = {}
    for i, n in enumerate(children_layer):
        children[n] = i

    if sparse:
        scope = lil_matrix((len(layer), len(children_layer)), dtype=bool)
    else:
        scope = np.zeros((len(layer), len(children_layer)), dtype=bool)

    for i, n in enumerate(layer):
        for c in n.children:
            scope[i, children[c]] = 1

    return scope


def get_two_layer_scopes(sum_layer, grand_children_layer, sparse):
    children = {}
    for i, n in enumerate(grand_children_layer):
        children[n] = i

    # print('children done')
    scopes = []

    for n in sum_layer:
        if sparse:
            scope = dok_matrix((len(n.children), len(grand_children_layer)), dtype=bool)
        else:
            scope = np.zeros((len(n.children), len(grand_children_layer)), dtype=bool)

        for i, c in enumerate(n.children):
            for g in c.children:
                scope[i, children[g]] = 1

        scopes.append(scope)

    return scopes


class Layer():
    def __init__(self, nodes):
        self.nodes = []
        self.nodes.extend(nodes)

    @property
    def n_nodes(self):
        return len(self.nodes)


class LeafLayer(Layer):
    def __init__(self, nodes):
        Layer.__init__(self, nodes)

        self.gaussian = []
        self.categorical = []

        for i, n in enumerate(nodes):
            if isinstance(n, Gaussian):
                self.gaussian.append((n.scope[0], n.mean, n.stdev))
            elif isinstance(n, Categorical):
                self.categorical.append((n.scope[0], n.p))

        self.gaussian = np.array(self.gaussian)
        self.categorical = np.array(self.categorical)


class ProductLayer(Layer):
    def __init__(self, nodes, scope_matrix):
        Layer.__init__(self, nodes)
        self.scope_matrix = scope_matrix


class SumLayer(Layer):
    def __init__(self, nodes, scope_matrix, weights):
        Layer.__init__(self, nodes, )
        self.scope_matrix = scope_matrix
        self.weights = weights


class SumProductLayer(Layer):
    def __init__(self, nodes, scope_matrices, weights):
        Layer.__init__(self, nodes, )
        self.scope_matrices = scope_matrices
        self.weights = weights


def to_layers(spn, sparse=True, copy=True):
    with elapsed_timer() as e:
        if copy:
            spn = Copy(spn)
        print('copy', e())
        spn = Prune(spn, contract_single_parents=False)
        print('prune', e())
        complete_layers([spn], type(spn))
        print('complete layers', e())
        node_layers = get_topological_order_layers(spn)
        print('topo search', e())
        print('nr layers', len(node_layers))

        layers = [LeafLayer(node_layers[0])]
        for i in tqdm(range(1, len(node_layers))):
            cur_layer = node_layers[i]
            prev_layer = node_layers[i - 1]
            scope = get_scope(cur_layer, prev_layer, sparse)

            if isinstance(cur_layer[0], Sum):
                weights = np.concatenate(list(map(lambda x: x.weights, cur_layer)))
                layers.append(SumLayer(cur_layer, scope, weights))
            else:
                layers.append(ProductLayer(cur_layer, scope))
        print('to layer objects', e())
        return layers


def to_compressed_layers(spn):
    with elapsed_timer() as e:
        spn = Copy(spn)
        print('copy', e())
        spn = Prune(spn, contract_single_parents=False)
        print('prune', e())
        complete_layers([spn], type(spn))
        print('complete layers', e())
        node_layers = get_topological_order_layers(spn)
        print('topo search', e())
        print('nr layers', len(node_layers))

        layers = [LeafLayer(node_layers[0])]
        for i in range(1, len(node_layers)):

            cur_layer = node_layers[i]
            prev_layer = node_layers[i - 1]

            cur_is_sum = isinstance(cur_layer[0], Sum)
            prev_is_prod = isinstance(prev_layer[0], Product)

            # print(i, cur_is_sum, prev_is_prod)
            if cur_is_sum:
                weights = list(map(lambda x: x.weights, cur_layer))

            if cur_is_sum and prev_is_prod:
                # build sp layer
                # remove prod from previous layer
                layers.pop()
                scopes = get_two_layer_scopes(cur_layer, node_layers[i - 2], True)
                layers.append(SumProductLayer(cur_layer, scopes, weights))
            else:
                scope = get_scope(cur_layer, prev_layer, True)
                if cur_is_sum:
                    layers.append(SumLayer(cur_layer, scope, weights))
                else:
                    layers.append(ProductLayer(cur_layer, scope))
        print('to layer objects', e())
        return layers
