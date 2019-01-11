"""
The structure file for the layerwise SPN which handles all
of the computation for the layerwise SPN.
"""

import torch
import numpy as np

from torch.autograd import Variable as Variable
from spn.gpu.PyTorch import Nodes
from spn.gpu.PyTorch import Edges
from torch import Tensor


class LayerwiseSPN(torch.nn.Module):
    """
    A LayerwiseSPN is a network that holds all of the information
    regarding a layerwise SPN. To make full utilization of the fast
    tensor operations in tensorflow or PyTorch, having matrix-matrix
    operations are significantly faster than performing operations
    on a tree/graph based structure on the CPU. This makes larger SPNs
    significantly more feasible.
    """

    def __init__(self, is_cuda=False):
        super(LayerwiseSPN, self).__init__()
        self.leaflist = []
        self.edgelist = []
        self.nodelist = []
        self.is_cuda = is_cuda

    def forward(self):
        for layer in self.leaflist:
            val = layer()

        for layer in self.nodelist:
            val = layer()

        return val

    def tensor_to_var(self, tensor, requires_grad=False):
        """
        Takes in a tensor and returns a PyTorch Variable
        :param tensor: the tensor that the variable will be wrapped around
        :param require_grad: If the pytorch variable will be backpropagated on
        :returns: A pytorch variable that wraps around the tensor
        """
        if self.is_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad)

    def tensor_to_param(self, tensor, requires_grad):
        """
        Takes in a tensor and returns a PyTorch parameter (will be used in param
        to keep track of the global parameter state of the spn)
        :param tensor: the tensor that the parameter will be wrapped around
        :param require_grad: If the pytorch variable will be backpropagated on
        :returns: A pytorch parameter that wraps around the tensor
        """
        if self.is_cuda:
            tensor = tensor.cuda()
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)

    def feed(self, variable_to_value={}, marginal_to_value={}):
        """
        :param variable_to_value:
        :param marginal_to_value:
        """
        for k in variable_to_value:
            k.feed_val(variable_to_value[k])
        for k in marginal_to_value:
            k.feed_marginalize_mask(marginal_to_value[k])

    def add_gaussian_node(self, mean, std, param):
        """
        Adding Gaussian Nodes with mean mean and standard deviation std
        to the SPN structure

        :param mean: Mean of the gaussian node(s) being added
        :param std: Std of the gaussian node(s).
        :param param: The parameter space that will be modified as a result of adding
        the gaussian node to the spn.
        :returns: The Gaussian nodes
        """

        # Create a torch parameter from the parameters (that will require gradient,
        # since they are learnable parameters)
        mean_param = self.tensor_to_param(torch.from_numpy(mean), requires_grad=True)
        logstd_param = self.tensor_to_param(torch.from_numpy(np.log(std)), requires_grad=True)

        # Create the Gaussian Nodes
        nodes = Nodes.GaussianNodes(is_cuda=self.is_cuda, mean=mean_param, logstd=logstd_param)

        # Adding the parameter set to the global parameter space
        param.add_param(mean_param)
        param.add_param(logstd_param)
        # Adding the leaf to the leaflist
        self.leaflist.append(nodes)
        return nodes

    def add_binary_nodes(self, num):
        """
        :param num: The number of binary nodes to be created
        """
        nodes = Nodes.BinaryNodes(is_cuda=self.is_cuda, num=num)
        self.leaflist.append(nodes)
        return nodes

    def add_sum_nodes(self, num):
        """
        :param num: The number of sum nodes to be created
        """
        nodes = Nodes.SumNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(nodes)
        return nodes

    def add_product_nodes(self, num):
        """
        :param num: The number of product nodes to be created
        """
        nodes = Nodes.ProductNodes(is_cuda=self.is_cuda, num=num)
        self.nodelist.append(nodes)
        return nodes

    def add_sum_node_weights(self, weights, parameters=None):
        """
        :param weights: The weights on the sum nodes (for now, must be positive,
        no renormalization of the weights occurs)
        :param parameters: The parameter set of the universe, will be updated
        as a result of this function.
        """
        assert parameters is not None
        weights = self.tensor_to_param(torch.from_numpy(weights), requires_grad=True)
        parameters.add_param(weights)

    def add_product_edges(self, lower_level, upper_level, connections=None):
        """
        :param lower_level: The lower level of the Sum Product Network when arranged
        topologically
        :param upper_level: The upper level of the Sum Product Network.
        :param connections: The connections between the upper level and the lower level
        represented as ones and zeroes for connections and no connections respectively
        """
        assert lower_level is not None
        assert upper_level is not None
        if connections is None:
            connections = self.tensor_to_var(torch.ones((lower_level.num, upper_level.num)))
        else:
            connections = self.tensor_to_var(torch.from_numpy(connections.astype("float32")))

        edges = Edges.ProductEdges(lower_level, upper_level, connections)
        upper_level.child_edges.append(edges)
        lower_level.parent_edges.append(edges)
        return edges

    def add_sum_edges(self, weights, lower_level, upper_level, connections=None, parameters=None):
        """
        :param weights: The weights on the sum edges.
        :param lower_level: The lower level of the Sum Product Network when arranged
        topologically
        :param upper_level: The upper level of the Sum Product Network.
        :param connections: The connections between the upper level and the lower level
        represented as ones and zeroes for connections and no connections respectively
        :param parameters: The parameter set of the universe, will be updated
        as a result of this function.
        """
        assert lower_level is not None
        assert upper_level is not None
        if connections is None:
            connections = self.tensor_to_var(torch.from_numpy(np.ones(weights.shape).astype("float32"))).detach()
        else:
            connections = self.tensor_to_var(torch.from_numpy(connections), requires_grad=False)
        weights = self.tensor_to_param(torch.from_numpy(weights), requires_grad=True)

        edges = Edges.SumEdges(lower_level, upper_level, connections, weights)
        upper_level.child_edges.append(edges)
        lower_level.parent_edges.append(edges)
        parameters.add_param(weights)

        return edges, weights

    def compute_unnormalized(self, variable_to_value=None, marginal_variables={}, log=False):
        """
        :param variable_to_value: Set of variables to their corresponding value
        :param marginal_variables: A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is marginalized (i.e., $X$)
            A mask of 0 indicates the variable is unmarginalized (i.e., $Y$)
        :return: a scalar which is the unnormalized measure of $\tilde{p}(Y | X)$
        """
        self.feed(variable_to_value, marginal_variables)
        if log:
            return self().data.cpu().numpy()
        else:
            return np.exp(self().data.cpu().numpy())

    def compute_probability(self, variable_to_value=None, conditional_mask=None, grad=False, log=False):
        """
        :param variable_to_value: Set of variables to their corresponding value
        :param conditional_mask:A dictionary containing <variable, mask> pairs
            A mask of 1 indicates the variable is conditioned (i.e., X)
            A mask of 0 indicates the variable is unconditioned (i.e., Y)
        :param grad: Whether the gradient is being computed or not
        :param log: Whether the log scale is active or not
        :return: a scalar, the normalized measure, $\tilde{p}(Y | X)$
        """
        log_p = self.compute_logunnormalized(variable_to_value)
        marginalize_mask = {}
        for k in conditional_mask:
            marginalize_mask[k] = 1 - conditional_mask[k]
        log_z = self.compute_logunnormalized(variable_to_value, marginalize_mask)
        negative_log_likelihood = torch.sum(-log_p + log_z)
        if grad:
            negative_log_likelihood.backward()

        prob = log_p.data.cpu().numpy() - log_z.data().cpu().numpy()
        if not log:
            prob = np.exp(prob)
        return prob
