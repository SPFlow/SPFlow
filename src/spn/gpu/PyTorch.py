import numpy as np
import torch
import torch.nn as nn

from spn.algorithms.TransformStructure import Copy
from spn.structure.Base import Product, Sum, eval_spn_bottom_up
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian


def log_prod_to_pytorch_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    tensor_sum = 0
    for child in children:
        tensor_sum += child
    return tensor_sum


def log_sum_to_pytorch_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    softmax_inverse = np.log(node.weights / np.max(node.weights)).astype(dtype)
    pytorch_weights = nn.s


def spn_to_pytorch_graph(node, data, batch_size=None, node_pytorch_graph, log_space=True, dtype=None):
    if not dtype:
        dtype = data.dtype

    variable_dict = {}
    # we don't need to create a placeholder in pytorch
    # instead we create a torch.tensor of the data shape
    data_placeholder = torch.zeros([batch_size, data.shape[1]], dtype=dtype)
    pytorch_graph = eval_spn_bottom_up(node=node, eval_functions=node_pytorch_graph, data_placeholder=data_placeholder,
                                       log_space=log_space, variable_dict=variable_dict, dtype=dtype)

    return pytorch_graph, data_placeholder, variable_dict


def eval_pytorch(spn, data, save_graph_path=None, dtype=np.float32):
    pytorch_graph, placeholder, _ = spn_to_pytorch_graph(spn, data, dtype=dtype)
    return eval_pytorch_graph(pytorch_graph, placeholder, data, save_graph_path)


def test_eval_gaussian():
    np.random.seed(17)
    data = np.random.normal(10, 0.01, size=2000).tolist() + \
        np.random.normal(30, 10, size=2000).tolist()
    data = np.array(data).reshape((-1, 10))
    data = data.astype(np.float32)

    ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1],
                         parametric_types=[Gaussian] * data.shape[1])
    spn = learn_parametric(data, ds_context)

    ll = log_likelihood(spn, data)

    # tf_ll = eval_pytorch(spn, data)

    self.assertTrue(np.all(np.isclose(ll, tf_ll)))


if __name__ == '__main__':
    test_eval_gaussian()
