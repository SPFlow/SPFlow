'''
This is a file that handles different nodes and their
forward passes in PyTorch
'''
import torch
import numpy as np
from torch.autograd import Variable as Variable


class Nodes(torch.nn.module):

    def __init__(self, is_cuda):
        super(Nodes, self).__init__()
        self.is_cuda = is_cuda

    def var(self, tensor, requires_grad=False):
        if self.is_cuda:
            tensor = tensor.cuda()

        return Variable(tensor, requires_grad)

class SumNodes(Nodes):

    def __init__(self, is_cuda, num=1):
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.is_cuda = is_cuda
        self.child_edges = []
        self.parent_edges = []

    def forward(self):
        batch = self.child_edges[0].child.val.size()[0]
        self.val = self.var(torch.zeros(batch, self.num))
        maxval = 0
        log_error_const = torch.exp(torch.FloatTensor([-75]))[0]
        if self.is_cuda:
            log_error_const = log_error_const.cuda()

        for i, e in enumerate(self.child_edges):
            temp_max = torch.max(torch.max(e.child.val))
            if i == 0:
                maxval = temp_max
            else:
                maxval = torch.max(temp_max, maxval)
        maxval.detach()

        for e in self.child_edges:
            temp = e.child.val - maxval
            temp = torch.exp(temp)
            true_weights = e.weights * e.mask
            self.val += torch.mm(temp, true_weights)


        self.val += log_error_const
        self.val = torch.log(self.val)
        self.val += maxval
        return self.val

class ProductNodes(Nodes):
    def __init__(self, is_cuda, num=1):
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.is_cuda = is_cuda
        self.child_edges = child_edges
        self.parent_edges = parent_edges

    def forward(self):
        batch = self.child_edges[0].child.val.size()[0]
        self.val = self.var(torch.zeros(batch, self.num))
        for e in self.child_edges:
            self.val += torch.mm(e.child_val, e.mask)
        return self.val


# Now the leaf nodes

class GaussianNodes(Nodes):
    def __init__(self, is_cuda, mean, logstd):
        Nodes.__init__(self, is_cuda).__init__()
        self.is_cuda = is_cuda
        self.mean = mean
        self.logstd = logstd
        self.parent_edges = []

    def feed_val(self, x, marginalize_connections=None):
        self.input = x
        batch = x.shape[0]

        if marginalize_connections is None:
            marginalize_connections = np.zeros((batch, self.num), dtype='float32')
        self.marginalize_connections = self.var(torch.from_numpy(marginalize_connections.astype('float32')))

    def feed_marginalize_connections(self, connections):
        self.marginalize_connections = self.var(torch.from_numpy(connections.astype('float32')))


    def forward(self):
        if isinstance(self.input, np.ndarray):
            self.input = torch.from_numpy(self.input.astype('float32'))
            self.input = self.var(self.input)

        x_mean = self.input - self.mean
        std = torch.exp(self.logstd)
        var = std*std

        self.val = (1 - self.marginalize_mask) * ( - (x_mean) * 
            (x_mean) / 2.0 / var - self.logstd - 0.91893853320467267)
        return self.val
