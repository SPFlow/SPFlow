from scipy.misc import logsumexp
from spn.algorithms import Inference

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Prune
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context, Product, Sum, get_depth, rebuild_scopes_bottom_up, get_topological_order_layers
import numpy as np

from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


np.random.seed(123)
train_data = np.c_[np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))],
                   np.r_[np.zeros((500, 1)), np.ones((500, 1))]]

spn_classification = learn_classifier(train_data,
                                      Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(
                                          train_data),
                                      learn_parametric, 2, min_instances_slice=100, cluster_univariate=True)


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


def get_scope(layer, children_layer):
    children = {}
    for i, n in enumerate(children_layer):
        children[n] = i

    scope = np.zeros((len(layer), len(children_layer)), dtype=bool)
    for i, n in enumerate(layer):
        for c in n.children:
            scope[i, children[c]] = 1

    return scope


class Layer():
    def __init__(self, nodes):
        self.nodes = []
        self.nodes.extend(nodes)


class LeafLayer(Layer):
    def __init__(self, nodes):
        Layer.__init__(self, nodes)


class ProductLayer(Layer):
    def __init__(self, nodes, scope_matrix):
        Layer.__init__(self, nodes)
        self.scope_matrix = scope_matrix


class SumLayer(Layer):
    def __init__(self, nodes, scope_matrix, weights):
        Layer.__init__(self, nodes, )
        self.scope_matrix = scope_matrix
        self.weights = weights


def to_layers(spn):
    spn = Prune(spn, contract_single_parents=False)

    complete_layers([spn], type(spn))

    node_layers = get_topological_order_layers(spn)

    layers = [LeafLayer(node_layers[0])]
    for i in range(1, len(node_layers)):
        cur_layer = node_layers[i]
        prev_layer = node_layers[i - 1]
        scope = get_scope(cur_layer, prev_layer)

        if isinstance(cur_layer[0], Sum):
            weights = np.concatenate(list(map(lambda x: x.weights, cur_layer)))
            layers.append(SumLayer(cur_layer, scope, weights))
        else:
            layers.append(ProductLayer(cur_layer, scope))

    return layers


def sum_lambda(layer, x):
    ll = np.empty((x.shape[0], layer.scope_matrix.shape[0]))
    for i, idx in enumerate(layer.scope_matrix):
        # ll[:, i] = logsumexp(x[:, idx], b=layer.nodes[i].weights, axis=1)
        # continue

        maxv = np.max(x[:, idx], axis=1, keepdims=True)
        np.einsum('ij,j->i', np.exp(x[:, idx] - maxv), layer.nodes[i].weights, out=ll[:, i])
        np.log(ll[:, i], out=ll[:, i])
        ll[:, i] += maxv[:, 0]

    return ll


def prod_lambda(layer, x):
    # ll = np.zeros((x.shape[0], len(layer.nodes)))
    ll = np.einsum('ij,kj->ik', x, layer.scope_matrix)
    return ll


def leaf_lambda(layer, data):
    res = np.zeros((data.shape[0], len(layer.nodes)))
    # return res
    l2p = 0.5 * np.log(2.0 * np.pi)
    with np.errstate(divide='ignore'):
        for i, n in enumerate(layer.nodes):
            if isinstance(n, Gaussian):
                res[:, i] = (data[:, n.scope[0]] - n.mean) / n.stdev
                res[:, i] = -np.log(n.stdev) - l2p - 0.5 * res[:, i] * res[:, i]
            elif isinstance(n, Categorical):
                np.log(np.array(n.p)[data[:, n.scope[0]].astype(int)], out=res[:, i])
            else:
                res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]
                # raise Exception('unknown dist')

            res[np.isnan(data[:, n.scope[0]]), i] = 1.0

    res[res == 0.0] = -0.0000000000001
    res[np.isinf(res)] = -200

    return res


default_layer_lambdas = {SumLayer: sum_lambda, ProductLayer: prod_lambda, LeafLayer: leaf_lambda}


def eval_layers(layers, data, layer_lambdas=default_layer_lambdas):
    x = data
    for layer in layers:
        x = layer_lambdas[type(layer)](layer, x)

    return x


import torch
import torch.nn as nn


class TorchLeavesLayer(nn.Module):
    def __init__(self, layer):
        super(TorchLeavesLayer, self).__init__()

        self.distributions = []
        self.scopes = []
        for n in layer.nodes:
            if isinstance(n, Categorical):
                self.distributions.append(torch.distributions.categorical.Categorical(torch.tensor(n.p).to('cuda')))
                self.scopes.append(n.scope[0])
            elif isinstance(n, Gaussian):
                self.distributions.append(
                    torch.distributions.normal.Normal(torch.tensor(n.mean).to('cuda'), torch.tensor(n.stdev).to('cuda')))
                self.scopes.append(n.scope[0])
            else:
                raise Exception("unknown dist")

    def forward(self, x):
        lls = torch.empty((x.shape[0], len(self.distributions)))
        for i in range(lls.shape[1]):
            val = x[:, self.scopes[i]]
            lls[:, i] = self.distributions[i].log_prob(val).float()
        return lls



class TorchProductLayer(nn.Module):
    def __init__(self, layer):
        super(TorchProductLayer, self).__init__()

        self.scope_matrix = torch.tensor(layer.scope_matrix).float()

    def forward(self, x):
        return torch.einsum('ij,kj->ik', x, self.scope_matrix)

        lls = torch.empty((x.shape[0], self.scope_matrix.shape[0]))
        return lls


class TorchSumLayer(nn.Module):
    def __init__(self, layer):
        super(TorchSumLayer, self).__init__()

        self.scope_matrix = torch.tensor(layer.scope_matrix).float()

        self.nodes = layer.scope_matrix.shape[0]
        self.weights = []
        self.idxs = []
        for i, idx in enumerate(layer.scope_matrix):
            self.weights.append(torch.log(torch.tensor(layer.nodes[i].weights)))
            self.idxs.append(torch.tensor(np.where(idx)))

    def forward(self, x):
        lls = torch.empty((x.shape[0], self.nodes))
        #return lls
        for i in range(self.nodes):
            y = x[:, self.idxs[i]] + self.weights[i]
            torch.logsumexp(y, dim=1, out=lls[:, i])
        return lls


layers = to_layers(spn_classification)

torchlayers = []
for l in layers:
    if isinstance(l, SumLayer):
        nl = TorchSumLayer(l)
    elif isinstance(l, ProductLayer):
        nl = TorchProductLayer(l)
    else:
        nl = TorchLeavesLayer(l)
    torchlayers.append(nl)

device = 'cuda'
spn = nn.Sequential(*torchlayers).to(device)
v = torch.from_numpy(train_data).float().to(device)

with elapsed_timer() as e:
    for _ in range(100):
        llold = log_likelihood(spn_classification, train_data)
    print('old', e())

with elapsed_timer() as e:
    for _ in range(100):
        ll = eval_layers(layers, train_data)
    print('new', e())


with elapsed_timer() as e:
    for _ in range(100):
        llp = spn(v)
    print('torch', e())

a = log_likelihood(spn_classification, train_data)
b = eval_layers(layers, train_data)
c = spn(v)
print('old', a[0], 'new', b[0], 'torch', c[0])
print("isclose new", np.all(np.isclose(a, b)))
print("isclose torch", np.all(np.isclose(a, c)))

0 / 0

ll = LeafLayerModule(layers[0])

with elapsed_timer() as e:
    for _ in range(1000):
        l0 = ll(v)
        l1 = torch_sum(layers[1], l0)
        l2 = torch_prod(layers[2], l1)
        l3 = torch_sum(layers[3], l2)
    print('torch', e())

0 / 0

with elapsed_timer() as e:
    for _ in range(100):
        llold = log_likelihood(spn_classification, train_data)
    print('old', e())
