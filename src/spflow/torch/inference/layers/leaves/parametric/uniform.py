"""
Created on August 17, 2022

@authors: Philipp Deibert
"""
import torch
import numpy as np
from typing import Optional
from spflow.meta.scope.scope import Scope
from spflow.meta.contexts.dispatch_context import DispatchContext, init_default_dispatch_context
from spflow.meta.dispatch.dispatch import dispatch
from spflow.torch.structure.layers.leaves.parametric.uniform import UniformLayer


@dispatch(memoize=True)
def log_likelihood(layer: UniformLayer, data: torch.Tensor, dispatch_ctx: Optional[DispatchContext]=None) -> torch.Tensor:
    """TODO"""
    # initialize dispatch context
    dispatch_ctx = init_default_dispatch_context(dispatch_ctx)

    batch_size: int = data.shape[0]

    # initialize empty tensor (number of output values matches batch_size)
    log_prob: torch.Tensor = torch.empty(batch_size, layer.n_out).to(layer.start.device)

    for node_id in range(layer.n_out):

        node_ids_tensor = torch.tensor([node_id])
        node_scope = layer.scopes_out[node_id]
        scope_data = data[:, node_scope.query]

        # ----- marginalization -----

        marg_mask = torch.isnan(scope_data).sum(dim=1) == len(node_scope.query)
        marg_ids = torch.where(marg_mask)[0]

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.meshgrid(marg_ids, node_ids_tensor, indexing='ij')] = 0.0

        # ----- log probabilities -----

        # create masked based on distribution's support
        valid_ids = layer.check_support(data[~marg_mask], node_ids=[node_id])

        # TODO: suppress checks
        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the TorchUniform distribution."
            )

        if layer.support_outside[node_id]:
            torch_valid_mask = torch.zeros(len(marg_mask), dtype=torch.bool)
            torch_valid_mask[~marg_mask] |= layer.dist(node_ids=[node_id]).support.check(scope_data[~marg_mask]).squeeze(1)
            
            outside_interval_ids = torch.where(~marg_mask & ~torch_valid_mask)[0]
            inside_interval_ids = torch.where(~marg_mask & torch_valid_mask)[0]

            # TODO: torch_valid_ids does not necessarily have the same dimension as marg_ids
            log_prob[torch.meshgrid(outside_interval_ids, node_ids_tensor, indexing='ij')] = -float("inf")

            # compute probabilities for values inside distribution support
            log_prob[torch.meshgrid(inside_interval_ids, node_ids_tensor, indexing='ij')] = layer.dist(node_ids=[node_id]).log_prob(
                scope_data[inside_interval_ids].type(torch.get_default_dtype())
            )
        else:
            # compute probabilities for values inside distribution support
            log_prob[~marg_mask] = layer.dist(node_ids=[node_id]).log_prob(
                scope_data[~marg_mask].type(torch.get_default_dtype())
            )
    
    return log_prob
    """
    # TODO: could be optimized, but difficult due to possibly different values for 'support_outside'
    # query rvs of all node scopes
    query_rvs = [list(set(scope.query)) for scope in layer.scopes_out]

    # group nodes by equal scopes
    for query_signature in np.unique(query_rvs, axis=0):

        # compute all nodes with this scope
        node_ids = np.where((query_rvs == query_signature).all(axis=1))[0].tolist()
        node_ids_tensor = torch.tensor(node_ids)
    
        # get data for scope (since all "nodes" are univariate, order does not matter)
        scope_data = data[:, layer.scopes_out[node_ids[0]].query]

        # ----- marginalization -----

        marg_mask = torch.isnan(scope_data).sum(dim=1) == len(query_signature)
        marg_ids = torch.where(marg_mask)[0]
        non_marg_ids = torch.where(~marg_mask)[0]

        # if the scope variables are fully marginalized over (NaNs) return probability 1 (0 in log-space)
        log_prob[torch.meshgrid(marg_ids, node_ids_tensor, indexing='ij')] = 0.0

        # ----- log probabilities -----
    
        # create masked based on distribution's support
        valid_ids = layer.check_support(data[~marg_ids])

        # TODO: suppress checks
        if not all(valid_ids):
            raise ValueError(
                f"Encountered data instances that are not in the support of the Uniform distribution."
            )
        
        # TODO: compute meshgrid once and reuse
        outside_interval_mask = torch.zeros((len(marg_mask), len(node_ids)), dtype=torch.bool)
        inside_interval_mask = torch.zeros((len(marg_mask), len(node_ids)), dtype=torch.bool)

        inside_interval_mask = layer.dist(node_ids=node_ids).support.check(scope_data[non_marg_ids, :])

        #outside_interval_mask[torch.meshgrid(non_marg_ids, node_ids_tensor, indexing='ij')] = valid
        #inside_interval_mask[torch.meshgrid(non_marg_ids, node_ids_tensor, indexing='ij')] = ~valid

        # to be able to compute log_probs in batch fashion, all data must lie within internval, due to torch distribution
        # therefore broacast scope data for each node and set values outside of corresponding distribution interval to respective start values
        starts = layer.start[node_ids_tensor]
        scope_data_broadcast = scope_data.repeat((1,len(node_ids)))
        scope_data_broadcast[~inside_interval_mask] = 0.0
        scope_data_broadcast += inside_interval_mask.int() * starts

        results = layer.dist(node_ids=node_ids).log_prob(
            scope_data[marg_ids, :].type(torch.get_default_dtype())
        )
        # set probabiliteis of values that were origianlly outside of interval to 0 (-infinity in log-space)
        results = 

        log_prob[torch.meshgrid(marg_ids, node_ids_tensor, indexing='ij')] = results

        # FUCK: what about gradients?
        log_pro

        # TODO: torch_valid_ids does not necessarily have the same dimension as marg_ids
        log_prob[outside_interval_mask] = -float("inf")

        # compute probabilities for values inside distribution support
        log_prob[inside_interval_mask] = layer.dist(node_ids=node_ids).log_prob(
            scope_data[torch_valid_mask].type(torch.get_default_dtype())
        )

    return log_prob
    """