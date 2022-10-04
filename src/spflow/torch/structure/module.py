"""
Created on June 10, 2021

@authors: Philipp Deibert

This file provides the abstract Module class for building graph structures with the PyTorch backend.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from spflow.meta.contexts.dispatch_context import DispatchContext
from spflow.meta.scope.scope import Scope
from spflow.meta.structure.module import MetaModule


class Module(MetaModule, nn.Module, ABC):
    """Abstract module class for building graph structures with the PyTorch backend.

    Attributes:
        children:
            List of child modules to form a directed graph of modules.
    """
    def __init__(self, children: Optional[List["Module"]]=None) -> None:

        super(Module, self).__init__()

        if children is None:
            children = []

        if any(not isinstance(child, Module) for child in children):
            raise ValueError("Children must all be of type 'Module'.")

        # register children
        for i, child in enumerate(children):
            self.add_module("child_{}".format(i + 1), child)

    def input_to_output_ids(self, input_ids: Union[List[int], torch.Tensor]) -> Tuple[List[int], List[int]]:

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
    
    @abstractmethod
    def n_out(self):
        pass


class NestedModule(Module, ABC):
    """Convenient module class for nesting non-terminal modules.
    
    Attributes:
        children:
            List of child modules to form a directed graph of modules.
    """
    def __init__(self, children: Optional[List[Module]]=None, **kwargs) -> None:
        """TODO"""
        if children is None:
            children = []
        
        super(NestedModule, self).__init__(children=children, **kwargs)
        self.placeholders = []

    def create_placeholder(self, input_ids: List[int]) -> "Placeholder":
        """Creates a placholder module that can be used for internal non-terminal modules.
        
        Also registers the placeholder internally.
        """
        # create and register placeholder
        ph = self.Placeholder(host=self, input_ids=input_ids)
        self.placeholders.append(ph)

        return ph
    
    def set_placeholders(self, f_name: str, inputs: torch.Tensor, dispatch_ctx: DispatchContext, overwrite=True) -> None:
        """Fills the cache for all registered placeholder modules given specified input values."""
        for ph in self.placeholders:
            # fill placeholder cache with specified input values
            dispatch_ctx.cache_value(f_name, ph, inputs[:, ph.input_ids], overwrite=overwrite)


    class Placeholder(Module):
        """Placeholder module as an intermediary module between nested non-terminal modules and actual child modules."""
        def __init__(self, host: Module, input_ids: List[int]) -> None:
            
            super().__init__()

            self.host = host
            self.input_ids = input_ids
            
            self.child_ids_actual, self.output_ids_actual = self.input_to_output_ids(list(range(len(input_ids))))

            # get child scopes
            child_scopes = sum([child.scopes_out for child in host.children()], [])
            
            # compute scope for placeholder
            self.scopes_out = [child_scopes[i] for i in input_ids]

        def input_to_output_ids(self, input_ids: List[int]) -> Tuple[int, int]:

            if len(input_ids) == 0:
                input_ids = list(range(len(self.input_ids)))
            
            return self.host.input_to_output_ids([self.input_ids[i] for i in input_ids])

        @property
        def n_out(self) -> int:
            return len(self.input_ids)