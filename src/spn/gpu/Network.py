'''
Constructing the Network in PyTorch's Backend
'''
import torch
from torch.autograd import Variable as Variable
import numpy as np

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import Nodes, Edges


class Network(torch.nn.Module):
    '''
    A particular SPN structure, whose parameters are in Param
    '''

    def __init__(self, is_cuda=False):
        '''
        Initialize the lists of nodes and edges
        '''
        super(Network, self).__init__()
        self.leaflist = list()  # Leaf list or dict
        self.nodelist = list()  # Variable list
        self.edgelist = list()  # Edgelist
        self.is_cuda = is_cuda

        return

    def forward(self):
        '''
        Override torch.nn.Module.forward
        :return: last layer's values
        '''
        for layer in self.leaflist:
            val = layer()

        for layer in self.nodelist:
            val = layer()

        return val

    def feed(self, val_dict={}, cond_mask_dict={}):
        '''
        Feed input values
        :param val_dict: A dictionary containing <variable, value> pairs
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned
        :return: None
        For variables not in `cond_dict`, assume not conditioned (i.e., query variables)
        For variables not in `val_dict`, assume to be marginalized out (i.e., all ones)
        '''

        for k in val_dict:
            k.feed_val(val_dict[k])
        for k in cond_mask_dict:
            k.feed_marginalize_mask(cond_mask_dict[k])

    def var(self, tensor, requires_grad=False):
        if self.is_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad)

    def parameter(self, tensor, requires_grad=False):
        if self.is_cuda:
            tensor = tensor.cuda()
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)

    def AddGaussianNodes(self, mean, std,
                         isReused=False,
                         parameters=None):
        '''
        Add a set of Bernoulli nodes
        :param mean: mean of Gaussian
        :param std:  std  of Gaussian
        :param isReused: if the parameters are reused
        :param param: the global parameter set of SPN
        :return: the Gaussian nodes
        For Node i, the probability of Node i being 1 is p_Bern[i].
        '''

        if isReused is None:
            print('not implemented')

        elif not isReused:
            mean = self.parameter(torch.from_numpy(mean), requires_grad=True)

            logstd = self.parameter(torch.from_numpy(np.log(std)), requires_grad=True)
        # else if isReused
        # do nothing

        _nodes = Nodes.GaussianNodes(is_cuda=self.is_cuda, mean=mean, logstd=logstd)

        if not isReused:
            parameters.add_param(mean, _nodes.mean_proj_hook)
            parameters.add_param(logstd, _nodes.std_proj_hook)
        self.leaflist.append(_nodes)
        return _nodes

    def AddSumEdges(self, lower, upper, weights=None, mask=None, isReused=False,
                    parameters=None):
        '''
        Add an edge: lower -> upper with weights weight
        :param lower: lower layer
        :param upper: upper layer
        :param weights: size N_lower x N_upper.
             `weights` is a numpy array if `isReused` is `False`
             `weights` is a instance of `torch.nn.Parameters` if `isReused` in `True`
             if `weights` is `None`, randomly inistalize weights according to some criterion
        :param mask: numpy aray with size <N_lower x N_upper> masking out non-connected edges (a mask of 0 indicates disconnected edge)
        :param isReused: Boolean. If is reused, the parameters are not added to `param`
        :param param: An instance of Param
        :return: (edge, para)
         edge is an instance of Edges.SumEdges
         para is an instance of torch.nn.Parameters
        '''

        if weights is None:
            # TODO: initialize weights according to some criterion
            pass
        elif not isReused:
            if mask is not None:
                mask = self.var(torch.from_numpy(mask), requires_grad=False)
            else:  # if mask is None:
                mask = self.var(torch.from_numpy(
                    np.ones(weights.shape).astype('float32'))).detach()

            weights = self.parameter(torch.from_numpy(weights), requires_grad=True)

        # else isReused:
        # weights are already an instance of `torch.nnParameters`, and have been in `param.para_list`
        # do nothing

        _edges = Edges.SumEdges(lower, upper, weights, mask)
        upper.child_edges.append(_edges)
        lower.parent_edges.append(_edges)

        if not isReused:
            parameters.add_param(weights, _edges.sum_weight_hook)

        return _edges, weights

    def AddSumNodes(self, num):
        '''
        Add a set of sum nodes to Network
        :param num: the number of sum nodes
        :return: An instance of Nodes.SumNodes
        '''
        _nodes = Nodes.SumNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddProductNodes(self, num):
        '''
        Add a set of product nodes to Network
        :param num: the number of product nodes
        :return: An instance of Nodes.ProductNodes
        '''
        _nodes = Nodes.ProductNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(_nodes)
        return _nodes

    def AddSumNodeWeights(self, weights, parameters=None):
        '''
        :param weights: the weights for this SPN
        '''
        self.weights = self.parameter(torch.from_numpy(weights), requires_grad=True)

        parameters.add_param(self.weights, hook=self.SumNodeWeightHook)

    def ComputeUnnormalized(self, val_dict=None, marginalize_dict={}):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)
        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''
        self.feed(val_dict, marginalize_dict)
        return np.exp(self().data.cpu().numpy())

    def ComputeLogUnnormalized(self, val_dict=None, marginalize_dict={}):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)
        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''

        self.feed(val_dict, marginalize_dict)

        return self()

    def ComputeProbability(self, val_dict=None, cond_mask_dict={}, grad=False,
                           log=False, is_negative=False):
        '''
        Compute unnormalized measure
        :param val_dict: A dictionary containing <variable, value> pairs
            X U Y = variables
        :param cond_mask_dict: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :return: a scalar, the unnormalized measure p_tilde(Y|X)
        For variables not in `cond_mask_dict`, assume not conditioned (i.e., Y)
        For variables not in `val_dict`, assumed to be marginalized out
        '''
        # TODO Not implemented: cond_mask_dict and/or val_dict not complete

        log_p_tilde = self.ComputeLogUnnormalized(val_dict)

        marginalize_dict = {}
        for k in cond_mask_dict:
            marginalize_dict[k] = 1 - cond_mask_dict[k]

        log_Z = self.ComputeLogUnnormalized(val_dict, marginalize_dict)

        J = torch.sum(- log_p_tilde + log_Z)  # negative log-likelihood

        if grad:
            J.backward()

        prob = log_p_tilde.data.cpu().numpy() - log_Z.data.cpu().numpy()
        if not log:
            prob = np.exp(prob)
        return prob
