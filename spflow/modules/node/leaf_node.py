"""Contains the basic abstract ``LeafNode`` module that all leaf nodes for SPFlow in the ``base`` backend.

All leaf nodes in the ``base`` backend should inherit from ``LeafNode`` or a subclass of it.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable

from spflow.modules.node.node import Node
from spflow.meta.data.scope import Scope


class LeafNode(Node, ABC):
    """Abstract base class for leaf nodes in the ``base`` backend.

    All valid SPFlow leaf nodes in the 'base' backend should inherit from this class or a subclass of it.

    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, scope: Scope, **kwargs) -> None:
        r"""Initializes ``LeafNode`` object.

        Args:
            scope:
                Scope object representing the scope of the leaf node,
        """
        super().__init__(children=[], **kwargs)

        # self.scope = scope
        self.scope = Scope([int(x) for x in scope.query], scope.evidence)

    @abstractmethod
    def accepts(self, signatures):
        """Checks if the leaf node accepts the given signatures.

        Args:
            signatures:
                List of FeatureContext objects representing the signatures to check.

        Returns:
            Boolean indicating if the leaf node accepts the given signatures.
        """
        pass

    @abstractmethod
    def from_signatures(self, signatures):
        """Creates a new leaf node from the given signatures.

        Args:
            signatures:
                List of FeatureContext objects representing the signatures to create the leaf node from.

        Returns:
            A new leaf node created from the given signatures.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Iterable:
        """Returns the parameters of the leaf node.

        Returns:
            Dictionary containing the parameters of the leaf node.
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> Iterable:
        """Returns the trainable parameters of the leaf node.

        Returns:
            Dictionary containing the training parameters of the leaf node.
        """
        pass
