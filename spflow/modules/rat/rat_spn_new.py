"""Contains the SPFlow architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``base`` backend.
"""

from typing import List, Optional, Union
from collections.abc import Iterable

from spflow.meta.dispatch import SamplingContext, init_default_sampling_context

from spflow.modules.module import Module

from spflow.modules.rat.region_graph import Partition, Region, RegionGraph
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.scope import Scope
from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.dispatch.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
)
from functools import partial
from typing import Any, Optional, Union
from collections.abc import Callable

import torch
from spflow import maximum_likelihood_estimation
from spflow.modules import Product
from spflow.modules import Sum
from spflow.modules import OuterProduct
from spflow.modules import ElementwiseProduct
from spflow.modules import ElementwiseSum
from spflow.modules.ops.cat import Cat
import numpy as np
import random
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.ops.split_halves import SplitHalves
from spflow.modules.ops.split_alternate import SplitAlternate
from spflow.modules.leaf.leaf_module import LeafModule
from spflow.modules.factorize import Factorize
from spflow.modules.rat_mixing_layer import MixingLayer


class RatSPN(Module):
    r"""Module architecture for Random and Tensorized Sum-Product Networks (RAT-SPNs) in the ``base`` backend.

    Constructs a RAT-SPN from a specified ``RegionGraph`` instance.
    For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

    Attributes:
        n_root_nodes:
            Integer specifying the number of sum nodes in the root region (C in the original paper).
        n_region_nodes:
            Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
        n_leaf_ndoes:
            Integer specifying the number of leaf nodes in each leaf region (I in the original paper).
        root_node:
            SPN-like sum node that represents the root of the model.
        root_region:
            SPN-like sum layer that represents the root region of the model.
    """

    def __init__(
            self,
            #region_graph: RegionGraph,
            leaf_modules: List[LeafModule],
            n_root_nodes: int,
            n_region_nodes: int,
            num_repetitions: int,
            depth: int,
            outer_product: Optional[bool] = False,
            num_splits: Optional[int] = 2,
    ) -> None:
        r"""Initializer for ``RatSPN`` object.

        Args:
            region_graph:
                ``RegionGraph`` instance to create RAT-SPN architecture from.
            feature_ctx:
                ``FeatureContext`` instance specifying the domains of the scopes.
                Scope must match the region graphs scope.
            n_root_nodes:
                Integer specifying the number of sum nodes in the root region (C in the original paper).
            n_region_nodes:
                Integer specifying the number of sum nodes in each (non-root) region (S in the original paper).
            n_leaf_ndoes:
                Integer specifying the number of leaf nodes in each leaf region (I in the original paper).

        Raises:
            ValueError: Invalid arguments.
        """
        super().__init__()
        self.n_root_nodes = n_root_nodes
        self.n_region_nodes = n_region_nodes
        self.n_leaf_nodes = leaf_modules[0].out_channels
        self.leaf_modules = leaf_modules
        self.depth = depth
        self.num_repetitions = num_repetitions
        self.outer_product = outer_product
        self.num_splits = num_splits

        if n_root_nodes < 1:
            raise ValueError(f"Specified value of 'n_root_nodes' must be at least 1, but is {n_root_nodes}.")
        if n_region_nodes < 1:
            raise ValueError(
                f"Specified value for 'n_region_nodes' must be at least 1, but is {n_region_nodes}."
            )
        if self.n_leaf_nodes < 1:
            raise ValueError(f"Specified value for 'n_leaf_nodes' must be at least 1, but is {self.n_leaf_nodes}.")

        if self.num_splits < 2:
            raise ValueError(f"Specified value for 'num_splits' must be at least 2, but is {self.num_splits}.")

        self.root_node = self.create_spn()


    def create_spn(self):

        if self.outer_product:
            product_layer = OuterProduct
        else:
            product_layer = ElementwiseProduct
        fac_layer = Factorize(inputs=self.leaf_modules, depth=self.depth, num_repetitions=self.num_repetitions)
        depth = self.depth
        root = None
        Split = SplitAlternate

        for i in range(depth):
            if i == 0 and depth > 1:
                out_prod = product_layer(inputs=Split(inputs=fac_layer, dim=1, num_splits=self.num_splits))
                sum_layer = Sum(inputs=out_prod, out_channels=self.n_region_nodes, num_repetitions=self.num_repetitions)
                root = sum_layer
            elif i == depth - 1:
                out_prod = product_layer(Split(inputs=root, dim=1, num_splits=self.num_splits))
                sum_layer = Sum(inputs=out_prod, out_channels=self.n_root_nodes, num_repetitions=self.num_repetitions)
                root = sum_layer
            else:
                out_prod = product_layer(Split(inputs=root, dim=1, num_splits=self.num_splits))
                sum_layer = Sum(inputs=out_prod, out_channels=self.n_region_nodes, num_repetitions=self.num_repetitions)
                root = sum_layer

        # MixtureLayer:
        #root = Sum(inputs=root, out_channels=1, num_repetitions=self.num_repetitions, sum_dim=3)
        root = MixingLayer(inputs=root, out_channels=1, num_repetitions=self.num_repetitions)

        return root


    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this module. Returns one since RAT-SPNs always have a single output."""
        return 1

    @property
    def feature_to_scope(self) -> list[Scope]:
        """Returns the mapping from features to scopes."""
        return self.root_node.feature_to_scope

    @property
    def scopes_out(self) -> list[Scope]:
        """Returns the output scopes of the RAT-SPN."""
        return self.root_node.scopes_out

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.root_node.to_dtype(dtype)

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def out_channels(self) -> int:
        return self._out_channels_total

    def to_device(self, device):
        if self.backend == "numpy":
            raise ValueError("it is not possible to change the device of models that have a numpy backend")
        self.device = device
        self.root_node.to_device(device)


@dispatch(memoize=True)  # type: ignore
def log_likelihood(
        rat_spn: RatSPN,
        data: torch.Tensor,
        check_support: bool = True,
        dispatch_ctx: Optional[DispatchContext] = None,
) -> torch.Tensor:
    """Computes log-likelihoods for RAT-SPNs nodes in the ``base`` backend given input data.

     Args:
         sum_node:
             Sum node to perform inference for.
         data:
             Two-dimensional NumPy array containing the input data.
             Each row corresponds to a sample.
         check_support:
             Boolean value indicating whether or not if the data is in the support of the leaf distributions.
             Defaults to True.
         dispatch_ctx:
             Optional dispatch context.

     Returns:
         Two-dimensional NumPy array containing the log-likelihoods of the input data for the sum node.
         Each row corresponds to an input sample.
     """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    ll = log_likelihood(
        rat_spn.root_node,
        data,
        check_support=check_support,
        dispatch_ctx=dispatch_ctx,
    )
    return ll


@dispatch  # type: ignore
def sample(
    rat_spn: RatSPN,
    data: torch.Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> torch.Tensor:

     dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
     sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    # call samples on specific repetition

     mixing_layer = rat_spn.root_node

     logits = mixing_layer.logits.unsqueeze(0).expand(
            sampling_ctx.channel_index.shape[0], -1, -1, -1, -1)

     if (
             "log_likelihood" in dispatch_ctx.cache
             and dispatch_ctx.cache["log_likelihood"][mixing_layer.inputs] is not None
     ):
         input_lls = dispatch_ctx.cache["log_likelihood"][mixing_layer.inputs]

         # Compute log posterior by reweighing logits with input lls
         log_prior = logits
         log_posterior = log_prior + input_lls.unsqueeze(3)
         log_posterior = log_posterior.log_softmax(dim=2)
         logits = log_posterior

     if is_mpe:
         # Take the argmax of the logits to obtain the most probable index
         sampling_ctx.repetition_idx = torch.argmax(logits, dim=-1)
     else:
         # Sample from categorical distribution defined by weights to obtain indices into input channels
         #sampling_ctx.repetition_idx = torch.distributions.Categorical(logits=logits.squeeze(-1)).sample().unsqueeze(-1)
         sampling_ctx.repetition_idx = torch.distributions.Categorical(logits=logits).sample().squeeze()

     return sample(
         rat_spn.root_node.inputs,
         data,
         is_mpe=is_mpe,
         check_support=check_support,
         dispatch_ctx=dispatch_ctx,
         sampling_ctx=sampling_ctx,
     )
