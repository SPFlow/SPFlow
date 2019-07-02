"""
Created on June 25, 2019

@author: Alejandro Molina
"""
import time
import numpy as np
from collections import defaultdict

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.TransformStructure import Copy
from spn.experiments.RandomSPNs.LearnRGSPN import Make_SPN_from_RegionGraph
from spn.experiments.RandomSPNs.region_graph import RegionGraph
from spn.structure.Base import get_topological_order_layers, Sum, Product, get_nodes_by_type
from spn.structure.leaves.parametric.Parametric import Gaussian
import math


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def get_params(ntype, nodes):
    if ntype == Gaussian:
        result = []

        for n in nodes:
            if isinstance(n, Gaussian):
                result.append((n.scope[0], n.mean, n.stdev, n.id))
            else:
                raise Exception("unkwnown node %s " % type(n))
        params = np.array(result)
    elif ntype == Product:
        result = []
        for node in nodes:
            if isinstance(node, Product):
                for c in node.children:
                    result.append((node.id, c.id))
            else:
                raise Exception("unkwnown node %s " % type(node))
        params = np.array(result)
    elif ntype == Sum:
        result = []
        for node in nodes:
            if isinstance(node, Sum):
                for i, c in enumerate(node.children):
                    result.append((node.id, c.id, node.weights[i]))
            else:
                raise Exception("unkwnown node %s " % type(node))
        params = np.array(result)
    else:
        raise Exception("unkwnown node %s " % type(ntype))
    return params


def Product_cuda(LL, params):
    instance_pos, param_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if param_pos >= params.shape[0]:
        return

    node_id = params[param_pos, 0]
    c_id = params[param_pos, 1]

    cuda.atomic.add(LL, (instance_pos, node_id), LL[instance_pos, c_id])


def Sum_cuda1(LL, params):
    instance_pos, param_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if param_pos >= params.shape[0]:
        return

    node_id = params[param_pos, 0]
    c_id = params[param_pos, 1]
    w = params[param_pos, 2]

    cuda.atomic.max(LL, (instance_pos, node_id), math.exp(LL[instance_pos, c_id]) * w)


def GaussianLeaf_cuda(LL, params, obs):
    instance_pos, param_pos = cuda.grid(2)

    if instance_pos >= LL.shape[0]:
        return
    if param_pos >= params.shape[0]:
        return

    # ps = params[param_pos]
    scope = int(params[param_pos, 0])
    mean = params[param_pos, 1]
    stdev = params[param_pos, 2]
    node_id = int(params[param_pos, 3])
    x = obs[instance_pos, scope]

    # k = 1/2 log(2*pi)
    k = np.float32(0.91893853320467274178032973640561763986139747363778341281)

    LL[instance_pos, node_id] = -math.log(stdev) - ((x - mean) ** 2.0 / (2.0 * stdev ** 2.0)) - k


mod = SourceModule(
    """
  __global__ void gaussian_leaf_cuda(float *LL, int llc, float *obs, int obsr, int obsc, float *params, int paramsr)
  {
    int idx = threadIdx.x + threadIdx.y * 4;
    if(idx > paramsr){
        return;
    }
    a[idx] *= 2;
  }
  """
)


def split_layers(layers):
    result = []

    for layer in layers:

        # split layer into different types
        node_types = defaultdict(list)
        for n in layer:
            node_types[type(n)].append(n)

        # add each type as a layer
        for node_type, v in node_types.items():
            node_ids = np.array(list(map(lambda n: n.id, v)))
            result.append((node_type, node_ids, get_params(node_type, v)))

    return result


if __name__ == "__main__":
    start = time.perf_counter()
    rg = RegionGraph(range(28 * 28))
    for _ in range(0, 2):
        # for _ in range(0, 20):
        rg.random_split(2, 2)

    rg_layers = rg.make_layers()
    print("random graph built in %s" % (time.perf_counter() - start))

    start = time.perf_counter()
    vector_list, root = Make_SPN_from_RegionGraph(
        rg_layers, np.random.RandomState(100), num_classes=1, num_gauss=20, num_sums=20
    )
    print("Make_SPN_from_RegionGraph in  %s" % (time.perf_counter() - start))

    start = time.perf_counter()
    print(get_structure_stats(root))
    print("get_structure_stats in  %s" % (time.perf_counter() - start))

    old_root = Copy(root)

    n_nodes = len(get_nodes_by_type(root))

    LL = np.zeros((10000, n_nodes), dtype=np.float32)
    X = np.random.randn(LL.shape[0] * LL.shape[1]).reshape((LL.shape[0], -1)).astype(np.float32)
    d_LL = cuda.mem_alloc(LL.nbytes)
    cuda.memcpy_htod(d_LL, LL)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(LL.shape[0] / threadsperblock[0])

    topo_layers = get_topological_order_layers(root)
    exec_layers = split_layers(topo_layers)

    d_LL = cuda.to_device(LL.astype(np.float32))
    d_X = cuda.to_device(X.astype(np.float32))

    for i in range(100):
        if i == 1:
            start = time.perf_counter()

        for layer_type, node_ids, params in exec_layers:
            d_params = cuda.to_device(params)

            if layer_type == Gaussian:
                blockspergrid_y = math.ceil(params.shape[0] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                GaussianLeaf_cuda[blockspergrid, threadsperblock](d_LL, d_params, d_X)

            elif layer_type == Product:
                blockspergrid_y = math.ceil(params.shape[0] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                # Product_cuda[blockspergrid, threadsperblock](d_LL, d_params)

            elif layer_type == Sum:
                blockspergrid_y = math.ceil(params.shape[0] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                # Sum_cuda[blockspergrid, threadsperblock](d_LL, d_params)

    print("Cuda LL in  %s" % (time.perf_counter() - start))
    result_array = d_LL.copy_to_host()

    pyll = np.zeros_like(LL)
    start = time.perf_counter()
    log_likelihood(root, X, lls_matrix=pyll)
    print("Python LL in  %s" % (time.perf_counter() - start))
