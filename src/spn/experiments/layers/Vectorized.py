import numpy as np
import torch
from scipy.special import logsumexp

from spn.algorithms import Inference
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Copy
from spn.experiments.layers.layers import SumLayer, ProductLayer, LeafLayer, to_layers, elapsed_timer, \
    to_compressed_layers, SumProductLayer
from spn.experiments.layers.pytorch import get_torch_spn, copy_parameters_back_from_torch_layers
from spn.structure.Base import Context, Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Bernoulli
# from numba import njit, prange
from timeit import timeit

np.random.seed(17)


def create_spflow_spn(n_feats, ctype=Gaussian):
    children1 = []
    children2 = []
    for i in range(n_feats):
        if ctype == Gaussian:
            c1 = Gaussian(np.random.randn(), np.random.rand(), scope=i)
            c2 = Gaussian(np.random.randn(), np.random.rand(), scope=i)
        else:
            #c1 = Bernoulli(p=1.0, scope=i)
            #c2 = Bernoulli(p=1.0, scope=i)
            c1 = Bernoulli(p=np.random.rand(), scope=i)
            c2 = Bernoulli(p=np.random.rand(), scope=i)

        children1.append(c1)
        children2.append(c2)

    prods1 = []
    prods2 = []
    for i in range(0, n_feats, 2):
        p1 = Product([children1[i], children1[i + 1]])
        p2 = Product([children2[i], children2[i + 1]])
        prods1.append(p1)
        prods2.append(p2)

    sums = []
    for i in range(n_feats // 2):
        s = Sum(weights=[0.5, 0.5], children=[prods1[i], prods2[i]])
        sums.append(s)

    spflow_spn = Product(sums)
    assign_ids(spflow_spn)
    rebuild_scopes_bottom_up(spflow_spn)
    return spflow_spn



def sum_prod_layer(layer, x):
    ll = np.empty((x.shape[0], layer.n_nodes))
    # return ll
    for i, scopes in enumerate(layer.scope_matrices):
        # continue
        pll = scopes * x.T
        # pll[np.isinf(pll)] = np.finfo(pll.dtype).min

        ll[:, i] = logsumexp(pll.T, b=layer.nodes[i].weights, axis=1)

    return ll


def sum_lambda(layer, x):
    ll = np.empty((x.shape[0], layer.n_nodes))
    # return ll
    for i, idx in enumerate(layer.scope_matrix):
        # continue
        ll[:, i] = logsumexp(x[:, idx.tocsr().indices], b=layer.nodes[i].weights, axis=1)

    return ll


def prod_lambda2(layer, x):
    # return prod(x, layer.n_nodes, layer.scope_matrix)
    # pll = np.einsum('ij,kj->ik', x, layer.scope_matrix)
    pll = x * layer.scope_matrix.T
    pll[np.isinf(pll)] = np.finfo(pll.dtype).min
    return pll
    return x * layer.scope_matrix.T
    # return np.matmul(x, layer.scope_matrix.T)
    # return np.empty((x.shape[0], layer.n_nodes))
    # ll = x * layer.scope_matrix.T
    # print(x.shape, layer.scope_matrix.shape)
    # ll = np.matmul(x, layer.scope_matrix.T)
    # ll = np.einsum('ij,kj->ik', x, layer.scope_matrix)
    # return ll


def leaf_lambda(layer, data):
    res = np.empty((data.shape[0], layer.n_nodes))
    # return res
    l2p = 0.5 * np.log(2.0 * np.pi)
    with np.errstate(divide='ignore'):
        for i, n in enumerate(layer.nodes):
            #res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]

            if isinstance(n, Gaussian):
                #res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]
                #continue

                res[:, i] = (data[:, n.scope[0]] - n.mean) / n.stdev
                res[:, i] = -np.log(n.stdev) - l2p - 0.5 * res[:, i] * res[:, i]

            elif isinstance(n, Categorical):
                np.log(np.array(n.p)[data[:, n.scope[0]].astype(int)], out=res[:, i])
            else:
                res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]
                # raise Exception('unknown dist')

            # res[np.isnan(data[:, n.scope[0]]), i] = 1.0

    # np.clip(res, -400, -0.0000000000001, out=res)

    # res[res == 0.0] = -0.0000000000001
    #res[np.isinf(res)] = np.finfo(res.dtype).min
    # print(res[0, :])
    return res


default_layer_lambdas = {SumProductLayer: sum_prod_layer, SumLayer: sum_lambda, ProductLayer: prod_lambda2,
                         LeafLayer: leaf_lambda}


def eval_layers(layers, x):
    for layer in layers:
        x = default_layer_lambdas[type(layer)](layer, x)

    return x


if __name__ == '__main__':
    device = 'cpu'

    ctype = Bernoulli

    nf = 1024*1
    spn_classification = create_spflow_spn(nf, ctype=ctype)


    #spn_classification = Product()
    #for i in range(nf):
    #    spn_classification.children.append(Bernoulli(p=1.0, scope=i))
    #assign_ids(spn_classification)
    #rebuild_scopes_bottom_up(spn_classification)

    train_data = np.random.rand(256, nf).astype(np.float32)

    if ctype == ctype:
        train_data = np.round(train_data)
        #train_data = np.ones_like(train_data)

    v = torch.from_numpy(train_data).to(device)

    clayers = to_compressed_layers(spn_classification)
    cspn = get_torch_spn(clayers).to(device)
    #cspn(v)

    spn_layers = to_layers(spn_classification)
    spn = get_torch_spn(spn_layers).to(device)
    copy_parameters_back_from_torch_layers(spn, spn_layers)

    layers = to_layers(spn_classification, sparse=True)

    bntimes = 10

    print('old', timeit(lambda: log_likelihood(spn_classification, train_data), number=bntimes))
    print('new', timeit(lambda: eval_layers(layers, train_data), number=bntimes))
    print('new compressed', timeit(lambda: eval_layers(clayers, train_data), number=bntimes))
    print('torch', timeit(lambda: spn(v), number=bntimes))
    print('torch compressed', timeit(lambda: cspn(v), number=bntimes))

    a = log_likelihood(spn_classification, train_data)
    b = eval_layers(layers, train_data)
    b2 = eval_layers(clayers, train_data)
    c = spn(v).detach().cpu().numpy()
    c2 = cspn(v).detach().cpu().numpy()
    la = log_likelihood(layers[-1].nodes[0], train_data)
    print('old', a[1], 'old layers', la[1], 'new', b[1], 'new compressed', b2[1], 'torch', c[1], 'torch compressed',
          c2[1])
    print("isclose layers", np.all(np.isclose(a, la)), np.sum(la))
    print("isclose new", np.all(np.isclose(a, b)), np.sum(b))
    print("isclose new compressed", np.all(np.isclose(a, b2)), np.sum(b2))
    print("isclose torch", np.all(np.isclose(a, c)), np.sum(c))
    print("isclose torch compressed", np.all(np.isclose(a, c2)), np.sum(c2))

    #now binary
