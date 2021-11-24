"""
Created on August 09, 2021

@authors: Kevin Huy Nguyen

This file provides the sampling methods for SPNs.
"""

from multipledispatch import dispatch  # type: ignore
from spflow.base.structure.nodes.node import (
    ILeafNode,
    IProductNode,
    ISumNode,
    INode,
    eval_spn_top_down,
)
from spflow.base.structure.network_type import SPN, NetworkType
from spflow.base.structure.nodes.validity_checks import _isvalid_spn
from spflow.base.sampling.nodes.leaves.parametric.sampling import sample_parametric_node
from spflow.base.inference.nodes.node import log_likelihood
import numpy as np
from numpy.random.mtrand import RandomState  # type: ignore
from typing import List, Callable, Type, Dict, Union


def sample_prod(
    node: ISumNode,
    input_vals: List,
    data: np.ndarray,
    rand_gen: np.random.RandomState,
) -> Dict[INode, np.ndarray]:
    """
    Sampling procedure for ProdNode. Passes on the input_vals to its children.

    Args:
        node:
            ProdNode to run sampling procedure on.
        input_vals:
            Determines which sample row in data should be sampled for a leafs rv.
        data:
            Data given to specify evidence or instances to be sampled by setting np.nan.
        rand_gen:
            Seed to specify random state.
    Returns: Dictionary, which has child nodes of input node as key and an array of integers,
             indicating which instances are to be sampled by which child.
    """

    conc_input_vals: np.ndarray = np.concatenate(input_vals)
    children_row_ids: Dict[INode, np.ndarray] = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = conc_input_vals
    return children_row_ids


def sample_sum(
    node: ISumNode,
    input_vals: List,
    data: np.ndarray,
    rand_gen: np.random.RandomState,
) -> Dict[INode, np.ndarray]:
    """
    Sampling procedure for sum nodes. Decides which child branch is to be sampled from by adding
    log-likelihood of child weights with drawn samples from gumbels distribution and then
    choosing which child is more likely according to the resulting sum.

    Args:
        node:
            Sum node to run sampling procedure on.
        input_vals:
            Determines which sample row in data should be sampled for this leafs rv.
        data:
            Data given to specify evidence or instances to be sampled by setting np.nan.
        rand_gen:
            Seed to specify random state.

    Returns: Dictionary, which has child nodes of input node as key and an array of integers,
             indicating which instances are to be sampled by which child.
    """

    conc_input_vals: np.ndarray = np.concatenate(input_vals)

    w_children_log_probs: np.ndarray = np.zeros((len(conc_input_vals), len(node.weights)))
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = np.log(node.weights[i])

    z_gumbels: np.ndarray = rand_gen.gumbel(
        loc=0,
        scale=1,
        size=(w_children_log_probs.shape[0], w_children_log_probs.shape[1]),
    )
    g_children_log_probs: np.ndarray = w_children_log_probs + z_gumbels
    rand_child_branches: Union[np.integer, np.ndarray] = np.argmax(g_children_log_probs, axis=1)

    children_row_ids: Dict[INode, np.ndarray] = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = conc_input_vals[rand_child_branches == i]

    return children_row_ids


def sample_leaf(
    node: ILeafNode,
    input_vals: List,
    data: np.ndarray,
    rand_gen: np.random.RandomState,
) -> None:
    """
    Samples leaf nodes according to their leaf type.

    Args:
        node:
            Leaf node to sample from.
        input_vals:
            Determines which sample instance in data should is sampled for this leafs rv.
        data:
            Data given to specify evidence or instances to be sampled by setting np.nan.
        rand_gen:
            Seed to specify random state.
    """

    conc_input_vals: np.ndarray = np.concatenate(input_vals)

    # find cells where nans are to be replaced with samples
    data_nans: np.ndarray = np.isnan(data[conc_input_vals, node.scope])

    n_samples: Union[np.number, np.ndarray] = np.sum(data_nans)

    if n_samples == 0:
        return None

    data[conc_input_vals[data_nans], node.scope] = sample_parametric_node(
        node, n_samples=n_samples, rand_gen=rand_gen
    )


_node_sampling: Dict[Type, Callable] = {
    IProductNode: sample_prod,
    ISumNode: sample_sum,
    ILeafNode: sample_leaf,
}


@dispatch(SPN, INode, np.ndarray, np.random.RandomState, _node_sampling=dict, in_place=bool)  # type: ignore[no-redef]
def sample_instances(
    network_type: NetworkType,
    node: INode,
    input_data: np.ndarray,
    rand_gen: np.random.RandomState,
    node_sampling: Dict[Type, Callable] = _node_sampling,
    in_place: bool = False,
) -> np.ndarray:
    """
    Samples instances according to nans specified in input_data given a SPN. By first
    doing a bottom-up pass to compute the likelihood taking into account the marginals.
    Then a top-down pass, to sample taking into account the likelihoods.

    Args:
        network_type:
            Specifies how to sample node.
        node:
            Root node of SPN to sample from.
        input_data:
            Data given to specify evidence or instances to be sampled by setting np.nan.
        rand_gen:
            Seed to specify random state.
        node_sampling:
            dictionary that contains k: Class of the node, v: lambda function that receives as
            parameters (node, input_vals, data, rand_gen)
        in_place:
            Boolean specifying whether to modify input_data in place or create copy of it.

    Returns: Sampled instances as np.ndarray.
    """

    if in_place:
        data: np.ndarray = input_data
    else:
        data = np.array(input_data)

    _isvalid_spn(node)

    assert np.all(
        np.any(np.isnan(data), axis=1)
    ), "each row must have at least a nan value where the samples will be substituted"

    log_likelihood(network_type, node, data)

    instance_ids: np.ndarray = np.arange(data.shape[0])

    eval_spn_top_down(node, node_sampling, parent_result=instance_ids, data=data, rand_gen=rand_gen)

    return data
