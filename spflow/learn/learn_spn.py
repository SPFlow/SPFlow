"""Contains the LearnSPN structure and parameter learner for SPFlow in the ``base`` backend."""

from collections.abc import Callable
from functools import partial
from itertools import combinations
from typing import Any

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from networkx import connected_components as ccnp, from_numpy_array

from spflow.meta.data.scope import Scope
from spflow.modules.base import Module
from spflow.modules.leaves.base import LeafModule
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.rdc import rdc


def prune_sums(node):
    """Prune unnecessary sum nodes from a probabilistic circuit.

    Recursively traverses the circuit and removes redundant sum nodes by flattening
    nested sum-cat structures and merging weights. Reduces circuit complexity while
    preserving the probability distribution.

    Args:
        node: Root node of the circuit to prune. Can be any module type,
            but pruning only affects Sum nodes.

    Returns:
        None: Modifies the circuit in-place.
    """
    if isinstance(node, Sum):
        child = node.inputs
        new_children = []
        new_weights = []
        if isinstance(child, Cat):
            # prune if all children of the cat module are sums
            all_sums = all(isinstance(c, Sum) for c in child.inputs)
            if all_sums:
                for j, c in enumerate(child.inputs):
                    new_children.append(c.inputs)
                    new_weights.append(c.weights)
        if len(new_children) != 0:
            # if we have new children, we need to update the weights
            current_weights = node.weights
            updated_weights = []
            for i in range(len(new_weights)):
                updated_weights.append(
                    new_weights[i] * current_weights[:, i, :].unsqueeze(1),
                )
            updated_weights = torch.concatenate(updated_weights, dim=1)

            all_cat = all(isinstance(c, Cat) for c in new_children)
            if all_cat:
                # if cat(cat) -> cat
                node.inputs = Cat([input_elem for c in new_children for input_elem in c.inputs], dim=2)
            else:
                node.inputs = Cat(new_children, dim=2)
            node.weights_shape = updated_weights.shape
            node.weights = updated_weights
            # call prune on the same node to prune in case new double sums are formed
            prune_sums(node)

        else:
            # call prune on the inputs if the children are not leaves modules
            if not isinstance(node.inputs, LeafModule):
                prune_sums(node.inputs)
    else:
        # call prune on the inputs
        if isinstance(node.inputs, Cat):
            prune_sums(node.inputs)
        else:
            if isinstance(node.inputs, torch.nn.ModuleList):
                for child in node.inputs:
                    # prune only if the child is not a leaves module
                    if not isinstance(child, LeafModule):
                        prune_sums(child)

def adapt_product_inputs(inputs: list[Module],leaf_oc, sum_oc) -> list[Module]:
    ref_oc = leaf_oc if leaf_oc > sum_oc else sum_oc
    output_modules = []
    for m in inputs:
        if m.out_channels < ref_oc:
            sum_module = Sum(inputs=m, out_channels=ref_oc)
            output_modules.append(sum_module)
        else:
            output_modules.append(m)
    return output_modules

def partition_by_rdc(
    data: torch.Tensor,
    threshold: float = 0.3,
    preprocessing: Callable | None = None,
) -> torch.Tensor:
    """Performs partitioning using randomized dependence coefficients (RDCs).

    Args:
        data: Two-dimensional Tensor containing the input data.
            Each row corresponds to a sample.
        threshold: Floating point value specifying the threshold for independence testing
            between two features. Defaults to 0.3.
        preprocessing: Optional callable that is called with ``data`` and returns another
            Tensor of the same shape. Defaults to None.

    Returns:
        One-dimensional Tensor with the same number of entries as the number of features
        in ``data``. Each integer value indicates the partition the corresponding feature
        is assigned to.
    """
    # perform optional pre-processing of data
    if preprocessing is not None:
        partitioning_data = preprocessing(data)
    else:
        partitioning_data = data

    # necessary for the correct precision for the rdc computation
    # Save original dtype and reset after computation to avoid contaminating global state
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)

        rdcs = torch.eye(data.shape[1], device=partitioning_data.device)
        for i, j in combinations(range(partitioning_data.shape[1]), 2):
            r = rdc(partitioning_data[:, i], partitioning_data[:, j])
            rdcs[j][i] = rdcs[i][j] = r

        # create adjacency matrix of features from thresholded rdcs
        rdcs[rdcs < threshold] = 0.0
        adj_mat = rdcs

        partition_ids = torch.zeros(data.shape[1], dtype=torch.int)

        for i, c in enumerate((ccnp(from_numpy_array(np.array(adj_mat.cpu().tolist()))))):
            partition_ids[list(c)] = i + 1

        return partition_ids.to(data.device)
    finally:
        torch.set_default_dtype(original_dtype)


def cluster_by_kmeans(
    data: torch.Tensor,
    n_clusters: int = 2,
    preprocessing: Callable | None = None,
) -> torch.Tensor:
    """Performs clustering using k-Means.

    Args:
        data: Two-dimensional Tensor containing the input data.
            Each row corresponds to a sample.
        n_clusters: Integer value specifying the number of clusters to be used.
            Defaults to 2.
        preprocessing: Optional callable that is called with ``data`` and returns another
            Tensor of the same shape. Defaults to None.

    Returns:
        One-dimensional Tensor with the same number of entries as the number of samples
        in ``data``. Each integer value indicates the cluster the corresponding sample
        is assigned to.
    """
    # perform optional pre-processing of data
    if preprocessing is not None:
        clustering_data = preprocessing(data)
    else:
        clustering_data = data

    kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", verbose=1)
    data_labels = kmeans.fit_predict(clustering_data)

    return data_labels


def learn_spn(
    data: torch.Tensor,
    leaf_modules: list[LeafModule] | LeafModule,
    out_channels: int = 1,
    min_features_slice: int = 2,
    min_instances_slice: int = 100,
    scope=None,
    clustering_method: str | Callable = "kmeans",
    partitioning_method: str | Callable = "rdc",
    clustering_args: dict[str, Any] | None = None,
    partitioning_args: dict[str, Any] | None = None,
    full_data: torch.Tensor | None = None,
) -> Module:
    """LearnSPN structure and parameter learner.

    LearnSPN algorithm as described in (Gens & Domingos, 2013): "Learning the Structure of Sum-Product Networks".

    Args:
        data: Two-dimensional Tensor containing the input data.
            Each row corresponds to a sample.
        leaf_modules: List of leaf modules or single leaf module to use for learning.
        out_channels: Number of output channels. Defaults to 1.
        min_features_slice: Minimum number of features required to partition.
            Defaults to 2.
        min_instances_slice: Minimum number of instances required to cluster.
            Defaults to 100.
        scope: Scope for the SPN. If None, inferred from leaf_modules.
        clustering_method: String or callable specifying the clustering method.
            If 'kmeans', k-Means clustering is used. If a callable, it should accept
            data and return cluster assignments.
        partitioning_method: String or callable specifying the partitioning method.
            If 'rdc', randomized dependence coefficients are used. If a callable, it
            should accept data and return partition assignments.
        clustering_args: Optional dictionary of keyword arguments for clustering method.
        partitioning_args: Optional dictionary of keyword arguments for partitioning method.
        full_data: Optional full dataset for parameter estimation.

    Returns:
        A Module representing the learned SPN.

    Raises:
        ValueError: If arguments are invalid or scopes are not disjoint.
    """
    if scope is None:
        if isinstance(leaf_modules, list):
            if len(leaf_modules) > 1:
                if not Scope.all_pairwise_disjoint([module.scope for module in leaf_modules]):
                    raise ValueError("Leaf modules must have disjoint scopes.")
                scope = leaf_modules[0].scope
                for leaf in leaf_modules[1:]:
                    scope = scope.join(leaf.scope)
            else:
                scope = leaf_modules[0].scope
        else:
            scope = leaf_modules.scope
            leaf_modules = [leaf_modules]

    # Verify that all indices in scope are valid for the data
    #if len(scope.query) > 0 and max(scope.query) >= data.shape[1]:
    #    raise ValueError(f"Scope indices {scope.query} exceed data features {data.shape[1]}.")

    # available off-the-shelf clustering methods provided by SPFlow
    if isinstance(clustering_method, str):
        # Randomized Dependence Coefficients (RDCs)
        if clustering_method == "kmeans":
            clustering_method = cluster_by_kmeans
        else:
            raise ValueError(f"Value '{clustering_method}' for partitioning method is invalid.")

    # available off-the-shelf partitioning methods provided by SPFlow
    if isinstance(partitioning_method, str):
        # Randomized Dependence Coefficients (RDCs)
        if partitioning_method == "rdc":
            partitioning_method = partition_by_rdc
        else:
            raise ValueError(f"Value '{partitioning_method}' for partitioning method is invalid.")

    # for convenience, directly bind additional keyword arguments to the methods
    if clustering_args is not None:
        clustering_method = partial(clustering_method, **clustering_args)
    if partitioning_args is not None:
        partitioning_method = partial(partitioning_method, **partitioning_args)

    if not isinstance(min_instances_slice, int) or min_instances_slice < 2:
        raise ValueError(
            f"Value for 'min_instances_slice' must be an integer greater than 1, but was: {min_instances_slice}."
        )
    if not isinstance(min_features_slice, int) or min_features_slice < 2:
        raise ValueError(
            f"Value for 'min_features_slice' must be an integer greater than 1, but was: {min_features_slice}."
        )

    def create_partitioned_mv_leaf(scope: Scope, data: torch.Tensor):
        """Create partitioned leaf nodes from scope and data.

        Creates leaf distributions by matching scope with available leaf modules and
        estimating parameters via maximum likelihood estimation.

        Args:
            scope: Variable scope defining which variables to create leaves for.
            data: Training data for parameter estimation.

        Returns:
            Union[Product, LeafModule]: Product node for multiple variables,
                or single leaf for univariate case.
        """
        leaves = []
        s = set(scope.query)
        for leaf_module in leaf_modules:
            leaf_scope = set(leaf_module.scope.query)
            scope_inter = s.intersection(leaf_scope)
            if len(scope_inter) > 0:
                leaf_layer = leaf_module.__class__(
                    scope=Scope(sorted(scope_inter)), out_channels=leaf_module.out_channels
                )
                # estimate leaves node parameters from data
                leaf_layer.maximum_likelihood_estimation(data)

                leaves.append(leaf_layer)

        if len(scope.query) > 1:
            if len(leaves) == 1:
                return Product(inputs=leaves[0])
            else:
                return Product(leaves)
        else:
            return leaves[0]

    # features does not need to be split any further
    if len(scope.query) < min_features_slice:
        return create_partitioned_mv_leaf(scope, data)

    else:
        # select correct data
        if not data.shape[0] == 1:
            partition_ids = partitioning_method(data[:, scope.query])  # uc

        # compute partitions of rvs from partition id labels
        partitions = []

        if not data.shape[0] == 1:
            for partition_id in torch.sort(torch.unique(partition_ids), axis=-1)[0]:  # uc
                partitions.append(torch.where(partition_ids == partition_id))  # uc

        # multiple partition (i.e., data can be partitioned)
        if len(partitions) > 1:
            product_inputs = []
            for partition in partitions:
                sub_structure = learn_spn(
                    data=data,
                    # data=data[:,partition[0]],  # TODO: check if this is correct -> seems not necessary since scope is passed
                    leaf_modules=leaf_modules,
                    scope=Scope([scope.query[rv] for rv in partition[0]]),
                    out_channels=out_channels,
                    clustering_method=clustering_method,
                    partitioning_method=partitioning_method,
                    min_features_slice=min_features_slice,
                    min_instances_slice=min_instances_slice,
                )
                product_inputs.append(sub_structure)
            leaf_oc = leaf_modules[0].out_channels if isinstance(leaf_modules, list) else leaf_modules.out_channels
            adapted_product_inputs = adapt_product_inputs(product_inputs, leaf_oc, out_channels)
            return Product(adapted_product_inputs)

        else:
            # if not enough instances to cluster, create leaves layer (can be set to prevent overfitting too much or to reduce network size)
            if data.shape[0] < min_instances_slice:
                return create_partitioned_mv_leaf(scope, data)
            # cluster data
            else:
                labels_per_channel = []
                # create cluster for each channel
                for i in range(out_channels):
                    labels = clustering_method(data)
                    labels_per_channel.append(labels)

                # non-conditional clusters
                if not scope.is_conditional():
                    sum_vectors = []
                    # create sum node for each channel
                    for labels in labels_per_channel:
                        inputs_per_channel = []

                        # Recurse for each label
                        #
                        # for each cluster, create a substructure
                        for cluster_id in torch.unique(labels):
                            sub_structure = learn_spn(
                                data[labels == cluster_id, :],
                                leaf_modules=leaf_modules,
                                scope=scope,
                                out_channels=out_channels,
                                clustering_method=clustering_method,
                                partitioning_method=partitioning_method,
                                min_features_slice=min_features_slice,
                                min_instances_slice=min_instances_slice,
                            )
                            inputs_per_channel.append(sub_structure)

                        # compute weights
                        w = []
                        for cluster_id in torch.unique(labels):
                            probs = torch.sum(labels == cluster_id) / data.shape[0]
                            w.append(probs)

                        weights = torch.tensor(w).unsqueeze(0).unsqueeze(-1)  # shape(1, num_clusters, 1)
                        if len(inputs_per_channel) == 1:
                            inputs = inputs_per_channel
                        else:
                            inputs = Cat(inputs_per_channel, dim=2)

                        weights_stack = []
                        for idx, child in enumerate(inputs_per_channel):
                            out_c = child.out_channels
                            weights_stack.append(weights[:, idx, :].repeat(out_c, 1) / out_c)

                        weights = (torch.cat(weights_stack)).unsqueeze(0)
                        sum_vectors.append(Sum(inputs=inputs, weights=weights))

                    if len(sum_vectors) == 1:
                        return sum_vectors[0]
                    else:
                        return Cat(sum_vectors, dim=2)

                # conditional clusters
                else:
                    raise NotImplementedError("Conditional clustering not yet implemented.")
                    pass
                    """
                    return CondSum(
                        children=[
                            learn_spn(
                                data[labels == cluster_id, :],
                                feature_ctx,
                                clustering_method=clustering_method,
                                partitioning_method=partitioning_method,
                                min_features_slice=min_features_slice,
                                min_instances_slice=min_instances_slice,
                            )
                            for cluster_id in torch.unique(labels)
                        ],
                    )
                    """
