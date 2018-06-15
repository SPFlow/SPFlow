'''
Created on June 10, 2018

@author: Alejandro Molina
'''
import time
from numba import cuda

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.TransformStructure import SPN_Reshape, Prune, Copy
from spn.algorithms.Validity import is_valid, has_valid_ids
from spn.experiments.RandomSPNs.LearnRGSPN import Make_SPN_from_RegionGraph
from spn.experiments.RandomSPNs.region_graph import RegionGraph
import numpy as np

from spn.structure.Base import Sum, get_nodes_by_type, Leaf, Product
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
from spn.structure.leaves.parametric.Parametric import Gaussian
import math


def get_execution_layers(spn):
    all_nodes = set(get_nodes_by_type(spn, ntype=(Sum, Product)))
    next_filter_type = Product
    leaves = get_nodes_by_type(spn, Leaf)
    layers = [np.asarray([n.id for n in leaves])]
    layer_types = [Leaf]
    seen_nodes = set(leaves)
    while len(all_nodes) > 0:
        filtered_nodes = []
        new_all_nodes = set()

        filter_type = next_filter_type
        for n in all_nodes:
            if isinstance(n, filter_type) and set(n.children).issubset(seen_nodes):
                filtered_nodes.append(n)
            else:
                new_all_nodes.add(n)

        if filter_type == Product:
            next_filter_type = Sum
        else:
            next_filter_type = Product

        if len(filtered_nodes) == 0:
            continue

        assert all_nodes == new_all_nodes | set(filtered_nodes)

        layer_types.append(filter_type)
        all_nodes = new_all_nodes
        layers.append(np.asarray([n.id for n in filtered_nodes]))
        seen_nodes.update(filtered_nodes)

    return layers, layer_types


def get_parameters(spn):
    val, err = has_valid_ids(spn)
    assert val, err

    all_nodes = get_nodes_by_type(spn)

    params = np.zeros((4, len(all_nodes)))

    for n in all_nodes:
        if isinstance(n, Gaussian):
            params[0, n.id] = n.scope[0]
            params[1, n.id] = n.mean
            params[2, n.id] = n.stdev
        else:
            assert len(n.children) == 2, "sum node with more than 2 children %s " % (len(n.children))
            params[0, n.id] = n.children[0].id
            params[1, n.id] = n.children[1].id
            if isinstance(n, Sum):
                params[2, n.id] = n.weights[0]
                params[3, n.id] = n.weights[1]

    return params


@cuda.jit
def GaussianLeaf_cuda(LL, params, node_ids, obs):
    instance_pos, node_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if node_pos >= node_ids.shape[0]:
        return

    node_id = node_ids[node_pos]
    scope = int(params[0, node_id])
    mean = params[1, node_id]
    stdev = params[2, node_id]
    x = obs[instance_pos, scope]

    # k = 1/2 log(2*pi)
    k = 0.91893853320467274178032973640561763986139747363778341281

    LL[instance_pos, node_id] = - math.log(stdev) - ((x - mean) ** 2 / (2.0 * stdev ** 2)) - k


@cuda.jit
def Product_cuda(LL, params, node_ids):
    instance_pos, node_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if node_pos >= node_ids.shape[0]:
        return

    node_id = node_ids[node_pos]
    left_id = int(params[0, node_id])
    right_id = int(params[1, node_id])

    LL[instance_pos, node_id] = LL[instance_pos, left_id] + LL[instance_pos, right_id]


@cuda.jit
def Sum_cuda(LL, params, node_ids):
    instance_pos, node_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if node_pos >= node_ids.shape[0]:
        return

    node_id = node_ids[node_pos]
    left_id = int(params[0, node_id])
    right_id = int(params[1, node_id])
    left_w = params[2, node_id]
    right_w = params[3, node_id]

    # log sum exp trick
    xleft = LL[instance_pos, left_id] + math.log(left_w)
    xright = LL[instance_pos, right_id] + math.log(right_w)

    xstar = max(xleft, xright)

    LL[instance_pos, node_id] = xstar + math.log(math.exp(xleft - xstar) + math.exp(xright - xstar))


if __name__ == '__main__':
    add_parametric_inference_support()

    start = time.perf_counter()
    rg = RegionGraph(range(28 * 28))
    for _ in range(0, 2):
        # for _ in range(0, 20):
        rg.random_split(2, 2)

    rg_layers = rg.make_layers()
    print("random graph built in  ", (time.perf_counter() - start))

    start = time.perf_counter()
    vector_list, root = Make_SPN_from_RegionGraph(rg_layers, np.random.RandomState(100),
                                                  num_classes=1, num_gauss=20, num_sums=20)
    print("Make_SPN_from_RegionGraph in  ", (time.perf_counter() - start))

    start = time.perf_counter()
    print(get_structure_stats(root))
    print("get_structure_stats in  ", (time.perf_counter() - start))

    old_root = Copy(root)

    start = time.perf_counter()
    root = Prune(root)
    print("Prune in  ", (time.perf_counter() - start))

    start = time.perf_counter()
    root = SPN_Reshape(root, 2)
    print("SPN_Reshape in  ", (time.perf_counter() - start))

    start = time.perf_counter()
    print(get_structure_stats(root))
    print("get_structure_stats in  ", (time.perf_counter() - start))

    start = time.perf_counter()
    layers, layer_types = get_execution_layers(root)
    print("get_execution_layers in  ", (time.perf_counter() - start))

    for i, lt in enumerate(layer_types):
        print(lt, len(layers[i]))

    max_id = max(map(lambda n: n.id, get_nodes_by_type(root)))
    print(max_id)

    children_sizes = list(map(lambda n: len(n.children), get_nodes_by_type(root, Sum)))
    print('cs ', np.unique(children_sizes, return_counts=True))
    params = sum(children_sizes)
    params += 2 * len(get_nodes_by_type(root, Leaf))

    start = time.perf_counter()
    params = get_parameters(root)
    print("get_parameters in  ", (time.perf_counter() - start))

    print(params)
    LL = np.zeros((100, params.shape[1]))
    X = np.random.randn(LL.shape[0] * LL.shape[1]).reshape((LL.shape[0], -1))

    lls_matrix = np.zeros_like(LL)

    print("LL size", lls_matrix.shape)

    print(" number of nodes ", len(get_nodes_by_type(root)))

    start = time.perf_counter()
    log_likelihood(root, X, lls_matrix=lls_matrix)
    print("it took in python ", (time.perf_counter() - start))

    start = time.perf_counter()

    d_LL = cuda.to_device(LL)
    d_params = cuda.to_device(params)
    d_X = cuda.to_device(X)

    for i, lt in enumerate(layer_types):
        # print(lt, len(layers[i]))
        node_ids = layers[i]
        # print(node_ids)

        # instance_pos, node_pos
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(LL.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(node_ids.shape[0] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        if lt == Leaf:
            GaussianLeaf_cuda[blockspergrid, threadsperblock](d_LL, d_params, node_ids, d_X)
        elif lt == Product:
            Product_cuda[blockspergrid, threadsperblock](d_LL, d_params, node_ids)
        else:
            Sum_cuda[blockspergrid, threadsperblock](d_LL, d_params, node_ids)

        # print(np.isclose(LL[:, node_ids], lls_matrix[:, node_ids]).all())
        # print("LL")
        # print(LL[:, node_ids])
        # print("pll")
        # print(lls_matrix[:, node_ids])
    d_LL.copy_to_host(LL)
    d_params.copy_to_host(params)

    end = time.perf_counter()
    print(np.isclose(LL, lls_matrix).all())
    print("it took in cuda ", (end - start))
