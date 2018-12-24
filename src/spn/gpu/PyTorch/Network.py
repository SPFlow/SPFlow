'''
The structure file for the layerwise SPN which handles all
of the computation for the layerwise SPN.
'''

import torch
import numpy as np

from torch.autograd import Variable as Variable

class LayerwiseSPN(torch.nn.module):
    '''
    A LayerwiseSPN is a network that holds all of the information
    regarding a layerwise SPN. To make full utilization of the fast
    tensor operations in tensorflow or PyTorch, having matrix-matrix
    operations are significantly faster than performing operations
    on a tree/graph based structure on the CPU. This makes larger SPNs
    significantly more feasible.
    '''
    def __init__(self, is_cuda=False):
        self.leaflist = []
        self.edgelist = []
        self.nodelist = []
        self.is_cuda = is_cuda

    def forward(self):
        for layer in self.leaflist:
            val = layer()

        for layer in self.nodelist:
            val = layer()

    def tensor_to_var(self, tensor, require_grad=False):
        '''
        Takes in a tensor and returns a PyTorch Variable
        :param tensor: the tensor that the variable will be wrapped around
        :param require_grad: If the pytorch variable will be backpropagated on
        :returns: A pytorch variable that wraps around the tensor
        '''
        if self.is_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, require_grad=require_grad)

    def tensor_to_param(self, tensor, require_grad):
        '''
        Takes in a tensor and returns a PyTorch parameter (will be used in param
        to keep track of the global parameter state of the spn)
        :param tensor: the tensor that the parameter will be wrapped around
        :param require_grad: If the pytorch variable will be backpropagated on
        :returns: A pytorch parameter that wraps around the tensor
        '''
        if self.is_cuda:
            tensor = tensor.cuda()
        return torch.nn.Parameter(tensor, require_grad=require_grad)

    def feed(self, variable_to_value, marginal_to_value):
        '''
        
        '''
        for k in variable_to_value:
            k.feed_val(variable_to_value[k])
        for k in marginal_to_value:
            k.feed_marginalize_mask(marginal_to_value[k])
        
    def add_gaussian_node(self, mean, std, param):
        '''
        Adding Gaussian Nodes with mean mean and standard deviation std
        to the SPN structure

        :param mean: Mean of the gaussian node(s) being added
        :param std: Std of the gaussian node(s).
        :param param: The parameter space that will be modified as a result of adding
        the gaussian node to the spn.
        :returns: The Gaussian nodes
        '''

        # Create a torch parameter from the parameters (that will require gradient, 
        # since they are learnable parameters)
        mean_param = self.parameter(torch.from_numpy(mean), require_grad=True)
        logstd_param = self.parameter(torch.from_numpy(np.log(std)), require_grad=True)
        
        # Create the Gaussian Nodes
        nodes = Nodes.GaussianNodes(is_cuda=self.is_cuda, mean=mean, logstd=logstd)

        # Adding the parameter set to the global parameter space
        param.add_param(mean)
        param.add_param(logstd)
        # Adding the leaf to the leaflist
        self.leaflist.append(nodes)
        return nodes

    def add_binary_nodes(self, num):
        nodes = Nodes.BinaryNodes(is_cuda=self.is_cuda, num=num)
        self.leaflist.append(nodes)
        return nodes

    def add_sum_nodes(self, num):
        nodes = Nodes.SumNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(nodes)
        return nodes

    def add_product_nodes(self, num):
        nodes = Nodes.ProductNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(nodes)
        return nodes

    def add_sum_node_weights(self, weights, parameters=None):
        assert(parameters != None)
        weights = self.parameter(torch.from_numpy(weights), require_grad=True)
        parameters.add_param(weights)

    def add_product_edges(self, lower_level, upper_level, connections=None):
        '''
        '''
        assert(lower_level != None)
        assert(upper_level != None)
        if connections == None:
            connections = self.var(torch.ones((lower_level.num, upper_level.num)))
        else:
            connections = self.var(torch.from_numpy(connections.astype('float32')))
            
        edges = Edges.ProductEdges(lower_level, upper_level, connections)
        upper_level.child_edges.append(edges)
        lower_level.child_edges.append(edges)
        return edges

    def add_sum_edges(self, weights, lower_level, upper_level, connections=None):
        '''

        '''
        assert(lower_level != None)
        assert(upper_level != None)
        if connections == None:
            connections = self.var(torch.from_numpy(np.ones(weights.shape).astype('float32'))).detach()
        else:
            connections = self.var(torch.from_numpy(connections), require_grad=False)
        weights = self.parameter(torch.from_numpy(weights, require_grad=True))

        edges = Edges.SumEdges(lower_level, upper_level, connections, weights)
        upper_level.child_edges.append(edges)
        lower_level.child_edges.append(edges)
        parameters.add_param(weights)

        return edges, weights

    def compute_unnormalized(self, variable_to_value=None, marginal_variables=None):
        self.feed(variable_to_value, marginal_variables)
        return np.exp(self().data.cpu().numpy())

    def compute_logunnormalized(self, variable_to_value=None, marginal_variables=None):
        self.feed(variable_to_value, marginal_variables)
        return self()

    def compute_probability(self, variable_to_value=None, conditional_mask=None, grad=False, log=False, is_negative=False):
        '''

        '''
        log_p = self.compute_logunnormalized(variable_to_value)
        marginalize_mask = {}
        for k in conditional_mask:
            marginalize_mask[k] = 1 - conditional_mask[k]
        log_z = self.compute_logunnormalized(val_dict, marginalize_mask)
        negative_log_likelihood = torch.sum(-log_p + log_z)
        prob = log_p.data.cpu().numpy() - log_z.data().cpu().numpy()
        return prob

    

