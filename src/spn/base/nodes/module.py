"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building nodes structures.
"""
from abc import ABC, abstractmethod


class Module(ABC):
    """Abstract module class for building nodes structures."""

    @abstractmethod
    def __len__(self):
        pass
