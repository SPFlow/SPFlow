"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures.
"""
from abc import ABC, abstractmethod
from spflow.base.structure.nodes.node import Node
from spflow.base.structure.network_type import NetworkType
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
        self.output_nodes: List[Node]
        self.children: Optional[List[Module]] = children

    @abstractmethod
    def __len__(self):
        pass
