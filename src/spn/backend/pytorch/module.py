"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures with the PyTorch backend.
"""
from abc import ABC, abstractmethod


class TorchModule(ABC):
    """Abstract module class for building graph structures with the PyTorch backend."""

    @abstractmethod
    def __len__(self):
        pass
