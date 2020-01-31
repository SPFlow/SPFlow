from itertools import chain, cycle

import torch
import torch.nn as nn

from spn.experiments.layers.layers import SumLayer, ProductLayer, SumProductLayer
from spn.experiments.layers.pytorch_parametric import MyBernoulli
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
import numpy as np
from tqdm import tqdm

_default_dist_lambdas = {
    "Categorical": lambda params: torch.distributions.categorical.Categorical(
        logits=torch.log(torch.clamp(params, 0.000000001, 0.999999999))),
    "Gaussian": lambda params: torch.distributions.normal.Normal(params[:, 0], torch.max(params[:, 1], torch.tensor(
        0.00001, device=params.device))),
    # "Bernoulli": lambda params: torch.distributions.bernoulli.Bernoulli(params[:, 0]),
    "Bernoulli": lambda params: MyBernoulli(params[:, 0]),
}


class TorchLeavesLayer(nn.Module):
    def __init__(self, layer, dist_lambdas=_default_dist_lambdas, no_grad=None):
        # no_grad should be a list of tuples
        super(TorchLeavesLayer, self).__init__()

        self.n_nodes = layer.n_nodes
        self.params = nn.ParameterList()
        self.dists = []  # (ntype, dist, scopes, idx)
        self.dist_lambdas = dist_lambdas

        self.tmp = {}
        self.tmp_no_grads = {}
        for idx, n in enumerate(layer.nodes):
            cur_dict = self.tmp if no_grad is not None and n.scope in no_grad else self.tmp_no_grads
            node_params = cur_dict.setdefault(type(n).__name__, [[], [], []])
            node_params[0].append(list(n.parameters))
            node_params[1].extend(n.scope)
            node_params[2].append(idx)

        for requires_grad, (ntype, node_params) in chain(zip(cycle([True]), self.tmp.items()),
                                                         zip(cycle([False]), self.tmp_no_grads.items())):
            params = nn.Parameter(torch.tensor(node_params[0]), requires_grad=requires_grad)
            self.params.append(params)
            dist = self.dist_lambdas[ntype](params)
            self.dists.append([ntype, dist, node_params[1], node_params[2]])

    def copy_params_back(self, spn_layer):
        pass

    def __init_dists(self):
        for i in range(len(self.dists)):
            ntype = self.dists[i][0]
            self.dists[i][1] = self.dist_lambdas[ntype](self.params[i])

    def _apply(self, fn):
        super(TorchLeavesLayer, self)._apply(fn)
        self.__init_dists()
        return self

    def forward(self, x):
        lls = torch.zeros((x.shape[0], self.n_nodes), device=x.device)
        # return lls
        for _, dist, scopes, idx in self.dists:
            val = x[:, scopes].squeeze(-1)
            nan_idx = torch.isnan(val)
            val[nan_idx] = 0  # assume value at pos 0 for nans

            lldist = dist.log_prob(val)  # .float()
            lldist[nan_idx] = 0  # marginalize
            lls[:, idx] = lldist

        # torch.clamp(lls, torch.finfo(lls.dtype).min, 0.0)
        # lls[lls == 0.0] = -0.0000000000001
        # lls[torch.isinf(lls)] = torch.finfo(lls.dtype).min

        return lls


class TorchSumProdLayer(nn.Module):
    def __init__(self, layer):
        super(TorchSumProdLayer, self).__init__()

        # self.scope_matrix = torch.tensor(layer.scope_matrix.todense()).float()

        self.n_nodes = layer.n_nodes
        self.weights = nn.ParameterList()
        self.sparse_scopes = nn.ParameterList()
        self.params = []
        self.scopes_out = []
        self.scopes_in = []
        for i, scope in enumerate(layer.scope_matrices):
            coo = scope.tocoo()

            if coo.data.size == 1:
                self.scopes_out.append(i)
                self.scopes_in.append(coo.data[0])
                continue

            values = torch.FloatTensor(coo.data)
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))

            sparse_scope = nn.Parameter(torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape)),
                                        requires_grad=False)

            weights = nn.Parameter(torch.log(torch.tensor(layer.weights[i])).unsqueeze(-1))
            self.weights.append(weights)
            weights_idx = len(self.weights) - 1

            self.sparse_scopes.append(sparse_scope)
            sparse_scopes_idx = len(self.sparse_scopes) - 1

            self.params.append((i, weights_idx, sparse_scopes_idx))

        self.scopes_out = torch.LongTensor(self.scopes_out)
        self.scopes_in = torch.LongTensor(self.scopes_in)

    def copy_params_back(self, spn_layer):
        for i, weights in enumerate(self.weights):
            w = torch.nn.functional.softmax(weights, dim=0)
            spn_layer.nodes[i].weights = w.detach().cpu().numpy().tolist()

    def forward(self, x):
        lls = torch.empty((x.shape[0], self.n_nodes), device=x.device)

        lls[:, self.scopes_out] = x[:, self.scopes_in]

        for (i, weights_idx, sparse_scopes_idx) in self.params:
            weights = self.weights[weights_idx]
            weights = torch.log(torch.nn.functional.softmax(weights, dim=0))

            sparse_scope = self.sparse_scopes[sparse_scopes_idx]

            y = torch.sparse.mm(sparse_scope, x.T) + weights
            # y = x * self.scopes[i] + self.weights[i]
            lls[:, i] = torch.logsumexp(y, dim=0)
        return lls


class TorchSumLayer(nn.Module):
    def __init__(self, layer):
        super(TorchSumLayer, self).__init__()

        # self.scope_matrix = torch.tensor(layer.scope_matrix.todense()).float()

        self.n_nodes = layer.scope_matrix.shape[0]
        self.weights = nn.ParameterList()
        self.idxs = []
        for i, idx in enumerate(layer.scope_matrix):
            self.weights.append(nn.Parameter(torch.log(torch.tensor(layer.nodes[i].weights))))
            self.idxs.append(torch.tensor(idx.tocsr().indices).long())

    def copy_params_back(self, spn_layer):
        for i, weights in enumerate(self.weights):
            w = torch.nn.functional.softmax(weights, dim=0)
            spn_layer.nodes[i].weights = w.detach().cpu().numpy().tolist()

    def forward(self, x):
        lls = torch.empty((x.shape[0], self.n_nodes), device=x.device)
        # return lls
        for i in range(self.n_nodes):
            weights = self.weights[i]
            weights = torch.log(torch.nn.functional.softmax(weights, dim=0))
            y = x[:, self.idxs[i]] + weights
            lls[:, i] = torch.logsumexp(y, dim=-1).squeeze(-1)
        return lls


class TorchProductLayer(nn.Module):
    def __init__(self, layer):
        super(TorchProductLayer, self).__init__()

        coo = layer.scope_matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        sparse_scope = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                                torch.Size(coo.shape))

        self.scope_matrix = nn.Parameter(sparse_scope, requires_grad=False)

    def copy_params_back(self, spn_layer):
        return

    def forward(self, x):
        # return torch.matmul(x, self.scope_matrix)
        pll = torch.sparse.mm(self.scope_matrix, x.T).T
        # pll = x * self.scope_matrix.T
        # pll = torch.einsum('ij,kj->ik', x, self.scope_matrix)
        # print(torch.any(torch.isinf(pll)))
        pll[torch.isinf(pll)] = torch.finfo(pll.dtype).min
        return pll


def get_torch_spn(layers):
    torchlayers = []
    for l in layers:
        if isinstance(l, SumLayer):
            nl = TorchSumLayer(l)
        elif isinstance(l, ProductLayer):
            nl = TorchProductLayer(l)
        elif isinstance(l, SumProductLayer):
            nl = TorchSumProdLayer(l)
        else:
            nl = TorchLeavesLayer(l)
        torchlayers.append(nl)
    # print(torchlayers)
    spn = nn.Sequential(*torchlayers)
    return spn


def copy_parameters_back_from_torch_layers(torch_layers, original_spn_layers):
    for i in range(len(torch_layers)):
        torch_layers[i].copy_params_back(original_spn_layers[i])
