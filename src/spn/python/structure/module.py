"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures.
"""
from abc import ABC, abstractmethod
from spn.python.structure.nodes.node import Node
from spn.python.structure.network_type import NetworkType
from typing import List, Optional


class Module(ABC):
    """Abstract module class for building graph structures.

    Attributes:
        children:
            List of child modules to form a graph of modules.
    """

    def __init__(self, children: Optional[List["Module"]]) -> None:
        self.root_node: Node
        self.nodes: List[Node]
        self.network_type: NetworkType
        self.output_nodes: Node
        self.children: Optional[List[Module]] = children

    @abstractmethod
    def __len__(self):
        pass
