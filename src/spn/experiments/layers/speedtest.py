import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6

from scipy import sparse
from scipy.misc import logsumexp

from spn.algorithms import Inference
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical

from spn.algorithms.Inference import log_likelihood

from spn.experiments.layers.Vectorized import to_layers, elapsed_timer, SumLayer, ProductLayer, LeafLayer
#from spn.experiments.layers.spflow_vs_pytorch import create_spflow_spn
import numpy as np
from scipy.sparse import find
import numexpr as ne


def to_sparse(layers):
    for l in layers:
        if hasattr(l, 'scope_matrix'):
            l.scope_matrix = sparse.csr_matrix(l.scope_matrix)




def sum_lambda(layer, x):
    ll = np.empty((x.shape[0], layer.scope_matrix.shape[0]))
    for i, idx in enumerate(layer.scope_matrix):
        ll[:, i] = logsumexp(x[:, idx.indices], b=layer.nodes[i].weights, axis=1)

    return ll


def prod_lambda(layer, x):
    # return np.zeros((x.shape[0], len(layer.nodes)))
    ll = x * layer.scope_matrix.T
    # ll = np.einsum('ij,kj->ik', x, layer.scope_matrix)
    return ll


def leaf_lambda(layer, data):
    res = np.zeros((data.shape[0], len(layer.nodes)))
    # return res
    l2p = 0.5 * np.log(2.0 * np.pi)
    with np.errstate(divide='ignore'):
        for i, n in enumerate(layer.nodes):
            # res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]
            # continue
            if isinstance(n, Gaussian):
                res[:, i] = (data[:, n.scope[0]] - n.mean) / n.stdev
                res[:, i] = -np.log(n.stdev) - l2p - 0.5 * res[:, i] * res[:, i]
                # mean = n.mean
                # stdev = n.stdev
                # val = data[:, n.scope[0]]
                # res[:, i] = ne.evaluate('-log(mean) - l2p - 0.5 * (((val - mean) / stdev) ** 2)')
            elif isinstance(n, Categorical):
                np.log(np.array(n.p)[data[:, n.scope[0]].astype(int)], out=res[:, i])
            else:
                res[:, i] = Inference._node_log_likelihood[n.__class__](n, data)[:, 0]
                # raise Exception('unknown dist')

            res[np.isnan(data[:, n.scope[0]]), i] = 1.0

    np.clip(res, -400, -0.0000000000001, out=res)
    # res[res == 0.0] = -0.0000000000001
    # res[np.isinf(res)] = -400
    # print(res[0, :])
    return res


default_layer_lambdas = {SumLayer: sum_lambda, ProductLayer: prod_lambda, LeafLayer: leaf_lambda}

import multiprocessing


def eval_layers2(layers, data, layer_lambdas=default_layer_lambdas):
    x = data
    for layer in layers:
        f = layer_lambdas[type(layer)]
        x = f(layer, x)

    return x


def eval_par(layers, data, cores=60):
    with multiprocessing.Pool(processes=cores) as pool:
        xi = np.array_split(data, cores)

        return np.concatenate(pool.starmap(eval_layers2, zip([layers], xi)), axis=1)


if __name__ == '__main__':


    nf = 2048
    spflow_spn = create_spflow_spn(nf)

    layers = to_layers(spflow_spn)

    to_sparse(layers)

    x = np.random.rand(1024 * 16, nf).astype(np.float32)

    log_likelihood(spflow_spn, x)
    eval_layers2(layers, x)

    with elapsed_timer() as e:
        for _ in range(1):
            log_likelihood(spflow_spn, x)
        print('old', e())

    with elapsed_timer() as e:
        for _ in range(1):
            # eval_layers(layers, x)
            eval_par(layers, x)
        print('new', e())

    print(spflow_spn)
