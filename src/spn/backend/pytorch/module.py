"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures with the PyTorch backend.
"""
from abc import ABC, abstractmethod
import torch.nn as nn


class TorchModule(ABC, nn.Module):
    """Abstract module class for building graph structures with the PyTorch backend."""

    @abstractmethod
    def __len__(self):
        pass
