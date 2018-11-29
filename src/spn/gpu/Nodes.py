import torch
from torch.autograd import Variable as Variable
import numpy as np


class Nodes(torch.nn.Module):
    '''
    Base class for SPN nodes (also called a layer).
    '''

    def __init__(self, is_cuda=False):
        '''
        Initialize nodes.
        :param is_cuda: True when computation should be done with CUDA (GPU).
        '''
        super(Nodes, self).__init__()

        self.is_cuda = is_cuda

    def var(self, tensor, requires_grad=False):
        '''
        Returns PyTorch Variable according to this node's settings.
        Currently only determines if the tensor is in GPU.
        :return: PyTorch Variable
        '''
        if self.is_cuda:
            tensor = tensor.cuda()

        return Variable(tensor, requires_grad)


class SumNodes(Nodes):
    '''
    The class of a set of sum nodes (also called a sum layer)
    '''

    def __init__(self, is_cuda, num=1):
        '''
        Initialize a set of sum nodes
        :param num: the number of sum nodes
        '''
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.child_edges = []
        self.parent_edges = []
        self.scope = None  # todo
        self.is_cuda = is_cuda

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of this layer
        '''
        batch = self.child_edges[0].child.val.size()[0]

        self.val = self.var(torch.zeros(batch, self.num))
        # with size: <1 x Ny>

        # we are about to subtract the maximum value in all child nodes
        # to make sure the exp is operated on non-positive values
        for idx, e in enumerate(self.child_edges):
            # TODO: be careful when donig batched computation
            tmpmax = torch.max(torch.max(e.child.val))
            if idx == 0:
                maxval = tmpmax
            else:
                maxval = torch.max(maxval, tmpmax)
        maxval.detach()  # disconnect during bp. any constant works here

        for e in self.child_edges:

            # e.child.val, size: <1 x Nx>
            # weights, size: <Nx x Ny>

            # log space computation:

            tmp = e.child.val - maxval  # log(x/max)
            # with size <1 x Nx>

            tmp = torch.exp(tmp)  # x/max
            # with size <1 x Nx>
            trueweights = e.weights * e.mask
            # with size <Nx x Ny>
            self.val += torch.mm(tmp, trueweights)  # <w, x>/max)
            # with size <1 x Ny>

            # original space
            # self.val += torch.mm(e.child.val, e.weights)

        # log space only:

        small_num = torch.exp(torch.FloatTensor([-75]))[0]
        if self.is_cuda:
            small_num = small_num.cuda()

        self.val += small_num
        self.val = torch.log(self.val)  # log( wx / max)
        self.val += maxval
        return self.val


class ProductNodes(Nodes):
    '''
    The class of a set of product nodes (also called a product layer)
    '''

    def __init__(self, is_cuda, num=1):
        '''
        Initialize a set of sum nodes
        :param num: the number of sum nodes
        '''
        Nodes.__init__(self, is_cuda).__init__()
        self.num = num
        self.child_edges = []
        self.parent_edges = []
        self.scope = None  # todo
        self.val = None
        self.samples = None

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of this layer
        '''
        # TODO: compute in the log space
        # however, we should first check the validity of log 0 for onehot leaf nodes
        batch = self.child_edges[0].child.val.size()[0]
        val = self.var(torch.zeros((batch, self.num)))
        # with size: batch x num_lower
        for e in self.child_edges:
            # log space
            val += torch.mm(e.child.val, e.mask)
            '''
            # original space
            num_child = e.child.num  # Is this variable going to be used?
            # child.val has size: <1 x Nx>
            # e.mask    has size: <Nx x Ny>  (suppose x -> y)
            tmp_val = torch.t(e.child.val.repeat(self.num, 1))
            # with size: <Nx x Ny>
            tmp_val = torch.pow(tmp_val, e.mask)
            # with size: <Nx x Ny>
            tmp_val = tmp_val.prod(0)
            # with size: <1 x Ny>
            val = val * tmp_val
            '''
        self.val = val
        return val


#######################
# Leaf nodes

class GaussianNodes(Nodes):
    '''
    The class of a set of Gaussian leaf nodes
    '''

    def __init__(self, is_cuda, mean, logstd):
        '''
        Initialize a set of Guassian nodes with parameters (mu, diag(sigma))
        :param mu:
        :param sigma:
        '''
        Nodes.__init__(self, is_cuda).__init__()

        self.num = 1          # the number of Gaussian nodes
        self.mean = mean
        self.logstd = logstd
        self.is_cuda = is_cuda
        self.parent_edges = []
        self.debug = False
        pass

    def forward(self):
        '''
        Overrides the method in torch.nn.Module
        :return: the value of current layer
        compute log p(x; mu, sigma)
        '''

        if isinstance(self.input, np.ndarray):
            self.input = torch.from_numpy(self.input.astype('float32'))
            self.input = self.var(self.input)

        x_mean = self.input - self.mean
        std = torch.exp(self.logstd)
        var = std * std

        self.val = (1 - self.marginalize_mask) * (- (x_mean) * (x_mean) /
                                                  2.0 / var - self.logstd - 0.91893853320467267)
        # Note: if marginalized out, log p = 0

        return self.val

    def feed_val(self, x, marginalize_mask=None):
        self.input = x
        batch = x.shape[0]
        if marginalize_mask is None:
            # do not marginalize
            marginalize_mask = np.zeros((batch, self.num), dtype='float32')
        # else if marginalize_mask specified:
            # do nothing
        self.marginalize_mask = self.var(torch.from_numpy(marginalize_mask.astype('float32')))
        pass

    def feed_marginalize_mask(self, mask):
        self.marginalize_mask = self.var(torch.from_numpy(mask.astype('float32')))

    def mean_proj_hook(self):
        pass

    def std_proj_hook(self):
        self.logstd.data = self.logstd.data.clamp(min=-85)
