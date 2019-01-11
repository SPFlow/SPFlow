import torch

"""
Keeps track of the parameter space of the SPN
"""


class Param(object):
    def __init__(self):
        self.parameter_list = torch.nn.ParameterList()
        # stores the parameters of the model in the form of a list
        """
        In this computation scheme, the model parameters are:
        1. Weights on the sum edges
        2. Parameters of the leaves
        """

    def add_param(self, parameter):
        """
        Adds a parameter to the parameter list.
        :param paramter: parameter to be appended to the parameter list of the
        model
        """

        # parameter must be of type torch.nn.parameter
        assert isinstance(parameter, torch.nn.Parameter)
        self.parameter_list.append(parameter)

    def __str__(self):
        ret = "\t".join(str(p) for p in self.parameter_list)
        return ret
