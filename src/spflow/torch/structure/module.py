# -*- coding: utf-8 -*-
"""Contains the abstract ``Module`` class for SPFlow modules in the 'torch' backend.

All valid SPFlow modules in the ``torch`` backend should inherit from this class or a subclass of it.
"""
from abc import ABC
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from spflow.meta.contexts.dispatch_context import DispatchContext
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
    def __init__(self, children: Optional[List["Module"]]=None) -> None:
        r"""Initializes ``Module`` object.

        Initializes module by correctly setting its children.

        Args:
            children:
                List of modules that are children to the module.

        Raises:
            ValueError: children of invalid type.
        """
        super(Module, self).__init__()

        if children is None:
            children = []

        if any(not isinstance(child, Module) for child in children):
            raise ValueError("Children must all be of type 'Module'.")

        # register children
        for i, child in enumerate(children):
            self.add_module("child_{}".format(i + 1), child)

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
        child_num_outputs = torch.tensor([child.n_out for child in self.children()])
        child_cum_outputs = torch.cumsum(child_num_outputs, dim=-1)

        # get child module for corresponding input
        child_ids = torch.sum(child_cum_outputs <= input_ids.reshape(-1,1), dim=1)
        # get output id of child module for corresponding input
        output_ids = (input_ids-(child_cum_outputs[child_ids.tolist()]-child_num_outputs[child_ids.tolist()]))

        # restore original shape
        child_ids = child_ids.reshape(shape)
        output_ids = output_ids.reshape(shape)

        return child_ids.tolist(), output_ids.tolist()


class NestedModule(Module, ABC):
    """Convenient abstract module class for modules in the ``torch`` backend that nest non-terminal modules.

    Methods:
        children():
            Iterator over all modules that are children to the module in a directed graph.

    Attributes:
        n_out:
            Integer indicating the number of outputs.
        scopes_out:
            List of scopes representing the output scopes.
    """
    def __init__(self, children: Optional[List[Module]]=None, **kwargs) -> None:
        """Initializes ``NestedModule`` object.

        Initializes module by correctly setting its children.

        Args:
            children:
                List of modules that are children to the module.
        """
        if children is None:
            children = []
        
        super(NestedModule, self).__init__(children=children, **kwargs)
        self.placeholders = []

    def create_placeholder(self, input_ids: List[int]) -> "Placeholder":
        """Creates an internal placeholder module.

        Creates and registers a placeholder module that can be used for internal non-terminal modules.

        Args:
            input_ids:
                List of integers specifying the input indices of the host module that this placeholder should represent internally.

        Returns:
            'Placeholder' object (to pass as child to internal non-terminal modules).
        """
        # create and register placeholder
        ph = self.Placeholder(host=self, input_ids=input_ids)
        self.placeholders.append(ph)

        return ph
    
    def set_placeholders(self, f_name: str, inputs: torch.Tensor, dispatch_ctx: DispatchContext, overwrite=True) -> None:
        """Fills the cache for all registered placeholder modules given a function name and specified input values.

        Args:
            f_name:
                String of the function name to set the cache of the placeholders for.
            inputs:
                PyTorch tensor of all inputs. Inputs to be cached are selected based on input indices the placeholders represent. 
            dispatch_ctx:
                Dispatch context to use cache of.
            overwrite:
                Boolean indicating whether or not to overwrite potentially existing cached values.
        """
        for ph in self.placeholders:
            # fill placeholder cache with specified input values
            dispatch_ctx.cache_value(f_name, ph, inputs[:, ph.input_ids], overwrite=overwrite)


    class Placeholder(Module):
        """Placeholder module as an intermediary module between nested non-terminal modules and actual child modules in the ``torch`` backend.

        Since all non-terminal modules need their children to be specified at creation, internal non-terminal modules would
        have to have the same children as the outer host module. This is not ideal, therefore placeholders can be used instead
        that simply act as mediators between the actual host module's children and the internal non-terminal modules.
        Furthermore, placeholders can be used to select parts of modules' outputs and use them internally in arbirary ways.

        Attributes:
            n_out:
                Integer indicating the number of outputs (equal to the number of inputs it represents).
            scopes_out:
                List of scopes representing the output scopes (equal to the scopes of the inputs it represents).
        """
        def __init__(self, host: Module, input_ids: List[int]) -> None:
            """Initializes ``Placeholder`` object.

            Initializes module by correctly setting its children.

            Args:
                host:
                    Host module, that the placeholder is part of.
                input_ids:
                    List of integers specifying the input indices of the host module that this placeholder should represent internally.
            """
            super().__init__()

            self.host = host
            self.input_ids = input_ids
            
            self.child_ids_actual, self.output_ids_actual = self.input_to_output_ids(list(range(len(input_ids))))

            # get child scopes
            child_scopes = sum([child.scopes_out for child in host.children()], [])
            
            # compute scope for placeholder
            self.scopes_out = [child_scopes[i] for i in input_ids]

        def input_to_output_ids(self, input_ids: Union[List[int], torch.Tensor]) -> Tuple[List[int], List[int]]:
            """Translates input indices to the host module into corresponding child module indices and child module output indices.

            For a given sequence of input indices to the host module (taking the inputs of all child modules into account), computes
            the corresponding child module indices and child module output indices.

            Args:
                input_ids:
                    List of integers or one-dimensional PyTorch tensor of integers specifying input indices to the module.

            Returns:
                A tuple of two lists of integers. The first list contains indices of child modules and the
                second list contains the corresponding output indices of the respective child modules.
            """
            if len(input_ids) == 0:
                input_ids = list(range(len(self.input_ids)))
            
            return self.host.input_to_output_ids([self.input_ids[i] for i in input_ids])

        @property
        def n_out(self) -> int:
            """Returns the number of output for this module."""
            return len(self.input_ids)