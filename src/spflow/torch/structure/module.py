"""Contains the abstract ``Module`` class for SPFlow modules in the 'torch' backend.

All valid SPFlow modules in the ``torch`` backend should inherit from this class or a subclass of it.
"""
from abc import ABC
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from spflow.meta.structure.module import MetaModule


class Module(MetaModule, nn.Module, ABC):
    r"""Abstract module class for building graph-based models in the ``torch`` backend.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, children: Optional[List["Module"]] = None) -> None:
        r"""Initializes ``Module`` object.

        Initializes module by correctly setting its children.

        Args:
            children:
                List of modules that are children to the module.

        Raises:
            ValueError: children of invalid type.
        """
        super().__init__()

        if children is None:
            children = []

        if any(not isinstance(child, Module) for child in children):
            raise ValueError("Children must all be of type 'Module'.")

        # register children as module list
        self.chs = nn.ModuleList(children)

    def input_to_output_ids(self, input_ids: Union[List[int], torch.Tensor]) -> Tuple[List[int], List[int]]:
        """Translates input indices into corresponding child module indices and child module output indices.

        For a given sequence of input indices (taking the inputs of all child modules into account), computes
        the corresponding child module indices and child module output indices.

        Args:
            input_ids:
                List of integers or one-dimensional PyTorch tensor of integers specifying input indices to the module.

        Returns:
            A tuple of two lists of integers. The first list contains indices of child modules and the
            second list contains the corresponding output indices of the respective child modules.
        """
        if len(input_ids) == 0:
            input_ids = list(range(self.n_out))

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        # remember original shape
        shape = input_ids.shape
        # flatten tensor
        input_ids = input_ids.ravel()

        # infer number of inputs from children (and their numbers of outputs)
        child_num_outputs = torch.tensor([child.n_out for child in self.chs])
        child_cum_outputs = torch.cumsum(child_num_outputs, dim=-1)

        # get child module for corresponding input
        child_ids = torch.sum(child_cum_outputs <= input_ids.reshape(-1, 1), dim=1)
        # get output id of child module for corresponding input
        output_ids = input_ids - (child_cum_outputs[child_ids.tolist()] - child_num_outputs[child_ids.tolist()])

        # restore original shape
        child_ids = child_ids.reshape(shape)
        output_ids = output_ids.reshape(shape)

        return child_ids.tolist(), output_ids.tolist()
