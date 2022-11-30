"""Contains the abstract ``Module`` class for SPFlow modules in the ``base`` backend.

All valid SPFlow modules in the ``base`` backend should inherit from this class or a subclass of it.
"""
from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np

from spflow.meta.structure.module import MetaModule


class Module(MetaModule, ABC):
    r"""Abstract module class for building graph-based models in the ``base`` backend.

    Attributes:
        children:
            List of modules that are children to the module in a directed graph.
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
        if children is None:
            children = []

        if any(not isinstance(child, Module) for child in children):
            raise ValueError("Children must all be of type 'Module'.")

        self.children = children

    def input_to_output_ids(
        self, input_ids: Union[List[int], np.ndarray]
    ) -> Tuple[List[int], List[int]]:
        """Translates input indices into corresponding child module indices and child module output indices.

        For a given sequence of input indices (taking the inputs of all child modules into account), computes
        the corresponding child module indices and child module output indices.

        Args:
            input_ids:
                List of integers or one-dimensional NumPy array of integers specifying input indices to the module.

        Returns:
            A tuple of two lists of integers. The first list contains indices of child modules and the
            second list contains the corresponding output indices of the respective child modules.
        """
        if len(input_ids) == 0:
            input_ids = list(range(self.n_out))

        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        # remember original shape
        shape = input_ids.shape
        # flatten tensor
        input_ids = input_ids.ravel()

        # infer number of inputs from children (and their numbers of outputs)
        child_num_outputs = np.array([child.n_out for child in self.children])
        child_cum_outputs = np.cumsum(child_num_outputs)

        # get child module for corresponding input
        child_ids = np.sum(
            child_cum_outputs <= input_ids.reshape(-1, 1), axis=1
        )
        # get output id of child module for corresponding input
        output_ids = input_ids - (
            child_cum_outputs[child_ids.tolist()]
            - child_num_outputs[child_ids.tolist()]
        )

        # restore original shape
        child_ids = child_ids.reshape(shape)
        output_ids = output_ids.reshape(shape)

        return child_ids.tolist(), output_ids.tolist()
