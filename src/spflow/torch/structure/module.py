"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures with the PyTorch backend.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import torch.nn as nn
from spflow.meta.structure.module import MetaModule


class Module(MetaModule, nn.Module, ABC):
    """Abstract module class for building graph structures with the PyTorch backend.

    Attributes:
        children:
            List of child modules to form a directed graph of modules.
    """
    def __init__(self, children: Optional[List["Module"]]) -> None:

        super(Module, self).__init__()

        if children is None:
            children = []

        if any(not isinstance(child, Module) for child in children):
            raise ValueError("Children must all be of type 'Module'.")

        # register children
        for i, child in enumerate(children):
            self.add_module("child_{}".format(i + 1), child)

    def input_to_output_id(self, input_ids: List[int]) -> List[Tuple[int, int]]:

        # infer number of inputs from children (and their numbers of outputs)
        child_num_outputs = [len(child) for child in self.children()]
        child_cum_outputs = np.cumsum(child_num_outputs)

        output_ids = []

        for input_id in input_ids:

            # get child module for corresponding input
            child_id = np.sum(child_cum_outputs <= input_id, axis=0).tolist()
            # get output id of child module for corresponding input
            output_id = input_id-(child_cum_outputs[child_id]-child_num_outputs[child_id])

            output_ids.append((child_id, output_id))

        return output_ids
    
    @abstractmethod
    def n_out(self):
        pass