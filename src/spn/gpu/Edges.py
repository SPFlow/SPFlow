import torch
import numpy as np

EPSILON = 0.00001


class ProductEdges():
    '''
    Handing product edges via masking and parent and children nodes
    '''

    def __init__(self, child, parent, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.mask = mask


class SumEdges():
    '''
    Handing the sum edges with the weights and the masking layers
    '''

    def __init__(self, child, parent, weights, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param weights: size: num_in x num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.weights = weights
        self.mask = mask

    def sum_weight_hook(self):
        if self.weights.size()[0] == 30:
            # pdb.set_trace()
            pass

        self.weights.data = self.weights.data.clamp(min=EPSILON)
