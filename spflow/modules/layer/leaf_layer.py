"""Contains the basic abstract ``LeafLayer`` module that all leaf nodes for SPFlow in the ``base`` backend.

All leaf nodes in the ``base`` backend should inherit from ``LeafLayer`` or a subclass of it.
"""
import numpy as np
from spflow.modules.node.leaf.utils import apply_nan_strategy
from spflow.modules.leaf_module import LeafModule
from spflow.meta.dispatch import SamplingContext
from torch import Tensor
from typing import Callable, Optional, Union
from spflow.meta.dispatch.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.sampling_context import init_default_sampling_context
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from spflow.meta.dispatch.dispatch import dispatch

from spflow.modules.node.node import Node
from spflow.meta.data.scope import Scope


class LeafLayer(LeafModule, ABC):
    """Abstract base class for leaf nodes in the ``base`` backend.

    All valid SPFlow leaf nodes in the 'base' backend should inherit from this class or a subclass of it.

    Attributes:
        n_out:
            Integer indicating the number of outputs. One for nodes.
        scopes_out:
            List of scopes representing the output scopes.
    """

    def __init__(self, scope: Scope, num_nodes_per_scope: int) -> None:
        r"""Initializes ``LeafLayer`` object.

        Args:
            scope: Scope object representing the scope of the leaf node,
            num_nodes: Number of nodes per scope.
        """
        event_shape = (len(scope.query), num_nodes_per_scope)
        super().__init__(scope=scope, event_shape=event_shape)

        self.num_nodes_per_scope = num_nodes_per_scope

    @property
    def n_out(self) -> int:
        """Returns the number of outputs for this node."""
        return np.prod(self.event_shape)


@dispatch  # type: ignore
def sample(
    leaf: LeafLayer,
    data: Tensor,
    is_mpe: bool = False,
    check_support: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
    sampling_ctx: Optional[SamplingContext] = None,
) -> Tensor:
    r"""Samples from the leaf nodes in the ``torch`` backend given potential evidence.

    Samples missing values proportionally to its probability distribution function (PDF).

    Args:
        leaf:
            Leaf node to sample from.
        data:
            Two-dimensional PyTorch tensor containing potential evidence.
            Each row corresponds to a sample.
        is_mpe:
            Boolean value indicating whether or not to perform maximum a posteriori estimation (MPE).
            Defaults to False.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.
        dispatch_ctx:
            Optional dispatch context.
        sampling_ctx:
            Optional sampling context containing the instances (i.e., rows) of ``data`` to fill with sampled values and the output indices of the node to sample from.

    Returns:
        Two-dimensional PyTorch tensor containing the sampled values together with the specified evidence.
        Each row corresponds to a sample.
    """
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)
    sampling_ctx = init_default_sampling_context(sampling_ctx, data.shape[0])

    if any([i >= data.shape[0] for i in sampling_ctx.instance_ids]):
        raise ValueError("Some instance ids are out of bounds for data tensor.")

    inverse_scope_query = list(filter(lambda x: x not in leaf.scope.query, range(data.shape[1])))
    # marg_ids = torch.isnan(data[:, leaf.scope.query])
    marg_ids = torch.isnan(data)
    marg_ids[:, inverse_scope_query] = False

    instance_ids_mask = torch.zeros(data.shape[0], 1, device=leaf.device, dtype=torch.bool)
    instance_ids_mask[sampling_ctx.instance_ids] = True

    sampling_mask = marg_ids & instance_ids_mask
    n_samples = torch.sum(sampling_mask.sum(1) > 0)  # count number of rows which have at least one true value
    if is_mpe:
        samples = leaf.distribution.mode()

        # Add batch dimension
        samples = samples.unsqueeze(0).repeat(n_samples, *([1] * (samples.dim())))
    else:
        samples = leaf.distribution.sample(n_samples=n_samples)

    # Use output_ids from sampling context to index into the correct outputs for each scope
    # I.e.: For each sample and for each
    assert samples.shape[0] == sampling_ctx.output_ids.shape[0]
    assert samples.shape[0] == data.shape[0]

    if leaf.event_shape[1] > 1:
        # Index the output_ids to get the correct samples for each scope
        # Output_ids should usually be defined by some module that is the parent of this layer
        assert samples.shape[1] == sampling_ctx.output_ids.shape[1]
        samples = samples.gather(dim=-1, index=sampling_ctx.output_ids.unsqueeze(-1)).squeeze(-1)

    # Set data at correct scope
    sampling_mask_at_scope = sampling_mask[:, leaf.scope.query]
    data[marg_ids] = samples[sampling_mask_at_scope]

    return data


@dispatch(memoize=True)  # type: ignore
def marginalize(
    layer: LeafLayer,
    marg_rvs: Iterable[int],
    prune: bool = True,
    dispatch_ctx: Optional[DispatchContext] = None,
) -> Optional[LeafLayer]:
    """Structural marginalization for ``NormallLayer`` objects in the ``torch`` backend.

    Structurally marginalizes the specified layer module.
    If the layer's scope contains none of the random variables to marginalize, then the layer is returned unaltered.
    If the layer's scope is fully marginalized over, then None is returned.

    Args:
        layer:
            Layer module to marginalize.
        marg_rvs:
            Iterable of integers representing the indices of the random variables to marginalize.
        prune:
            Boolean indicating whether or not to prune nodes and modules where possible.
            Has no effect here. Defaults to True.
        dispatch_ctx:
            Optional dispatch context.

    Returns:
        Unaltered leaf layer or None if it is completely marginalized.
    """
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    # Marginalized scope
    scope_marg = Scope([q for q in layer.scope.query if q not in marg_rvs])
    # Get indices of marginalized random variables in the original scope
    idxs_marg = [i for i, q in enumerate(layer.scope.query) if q in scope_marg.query]

    if len(scope_marg.query) == 0:
        return None

    # Construct new layer with marginalized scope and params
    marg_params_dict = layer.distribution.marginalized_params(idxs_marg)

    # Make sure to detach the parameters first
    marg_params_dict = {k: v.detach() for k, v in marg_params_dict.items()}

    # Construct new object of the same class as the layer
    return layer.__class__(
        scope=scope_marg,
        **marg_params_dict,
    )
