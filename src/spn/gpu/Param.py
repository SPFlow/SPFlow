'''
Handing the parameter space operations in SPFlow PyTorch Backend
'''

import torch
import numpy as np


class Param():
    '''
    Global status of the model parameters
    Currently keeps tracks of parameters only and allows us to perform
    operations on the PyTorch parameter space.
    TODO: Add more notes about the capabilities of the parameter space and how
     exactly to utilize it fully
    '''
    parameter_list = None  # Parameter list
    # refer to `https://pytorch.org/docs/stable/nn.html`

    def __init__(self):
        '''
        Initlize the parameter list
        '''
        self.parameter_list = torch.nn.ParameterList()
        self.hook_list = []

    def add_param(self, parameter, hook):
        '''
        Add a set of parameters to the world
        :param parameter: An instance of torch.nn.Parameters
        :param mask:
        :return: None
        If mask = None: do not normalize the corresponding parameter
        Else: normalize the parameter as a probabilistic distribution according to the
              variables indicated in the masks
        '''

        self.parameter_list.append(parameter)

        # hooks are a pytorch specific construct that can be called in the
        # forward and backward passes to allow for debugging through prints, etc

        if hook is not None:
            self.hook_list.append(hook)

    def register(self, model):
        for idx, parameter in enumerate(self.parameter_list):

            model.register_parameter('para_' + str(idx), parameter)
            self.model = model

    def get_unrolled_para(self):
        pvector = np.array([])

        for parameter in self.parameter_list:
            pvector = np.concatenate((pvector, parameter.data.numpy().reshape(-1)))
        return pvector.reshape((-1, 1))

    def get_unrolled_grad(self):
        pvector = np.array([])

        for parameter in self.parameter_list:
            pvector = np.concatenate((pvector, parameter.grad.data.numpy().reshape(-1)))

        return pvector.reshape((-1, 1))

    def get_size(self):
        '''
        :return: the shape of the parameter set
        '''
        return len(self.parameter_list)

    def set_shape(self, dims):
        '''
        :param dims: Dims is a tuple that has the parameters that determines the
        new shape of the parameter list
        :return: The new shape of the parameter list and modifies the actual
        dimensions of the parameter list
        '''
        self.parameter_list = np.array(self.parameter_list).reshape()


if __name__ == '__main__':

    param = Param()
