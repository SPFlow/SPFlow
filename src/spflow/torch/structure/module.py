"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures with the PyTorch backend.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn


class TorchModule(ABC, nn.Module):
    """Abstract module class for building graph structures with the PyTorch backend."""

    @abstractmethod
    def __len__(self):
        pass

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