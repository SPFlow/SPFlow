"""Contains the LearnSPN structure and parameter learner for SPFlow in the ``base`` backend.
"""
import time
from functools import partial
from typing import Any, Optional, Union
from collections.abc import Callable

from sklearn.cluster import KMeans
import torch
from spflow import maximum_likelihood_estimation
from spflow.modules import Product
from spflow.modules import Sum
from spflow.utils.kmeans import kmeans
from fast_pytorch_kmeans import KMeans
from spflow.modules.ops.cat import Cat
import numpy as np

from spflow.meta.data.scope import Scope
from spflow.modules.module import Module

from spflow.modules.leaf.leaf_module import LeafModule
from networkx import connected_components as ccnp, from_numpy_array
from spflow.utils.connected_components import connected_components
from spflow.utils.rdc import rdc
from itertools import combinations


def prune_sums(node):
    """
    iterates through spn and prunes unnecessary sums.
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
                updated_weights.append(new_weights[i] * current_weights[:, i, :].unsqueeze(1), )
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
            # call prune on the inputs if the children are not leaf modules
            if not isinstance(node.inputs, LeafModule):
                prune_sums(node.inputs)
    else:
        # call prune on the inputs
        if isinstance(node.inputs, Cat):
            prune_sums(node.inputs)
        else:
            if isinstance(node.inputs, torch.nn.ModuleList):
                for child in node.inputs:
                    # prune only if the child is not a leaf module
                    if not isinstance(child, LeafModule):
                        prune_sums(child)


def partition_by_rdc(
        data: torch.Tensor,
        threshold: float = 0.3,
        preprocessing: Optional[Callable] = None,
) -> torch.Tensor:
    """Performs partitioning using randomized dependence coefficients (RDCs) to be used with the LearnSPN algorithm in the ``base`` backend.

    Args:
        data:
            torchwo-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        threshold:
            Floating point value specifying the threshold for independence testing between two features.
            Defaults to 0.3.
        preprocessing:
            Optional callable that is called with ``data`` and returns another NumPy array of the same shape.
            Defaults to None.

    Returns:
        One-dimensional NumPy array with the same number of entries as the number of features in ``data``.
        Each integer value indicates the partition the corresponding feature is assigned to.
    """

    # perform optional pre-processing of data
    if preprocessing is not None:
        partitioning_data = preprocessing(data)
    else:
        partitioning_data = data

    # necessary for the correct precision fo thr rdc computation
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


def cluster_by_kmeans(
        data: torch.Tensor,
        n_clusters: int = 2,
        preprocessing: Optional[Callable] = None,
) -> torch.Tensor:
    """Performs clustering usig k-Means to be used with the LearnSPN algorithm in the ``base`` backend.

    Args:
        data:
            torchwo-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        n_clusters:
            Integer value specifying the number of clusters to be used.
            Defaults to 2.
        preprocessing:
            Optional callable that is called with ``data`` and returns another NumPy array of the same shape.
            Defaults to None.

    Returns:
        One-dimensional NumPy array with the same number of entries as the number of samples in ``data``.
        Each integer value indicates the cluster the corresponding sample is assigned to.
    """
    # perform optional pre-processing of data
    if preprocessing is not None:
        clustering_data = preprocessing(data)
    else:
        clustering_data = data

    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
    data_labels = kmeans.fit_predict(clustering_data)

    return data_labels


def learn_spn(
        data: torch.Tensor,
        leaf_modules: Union[list[LeafModule], LeafModule],
        out_channels: int = 1,
        min_features_slice: int = 2,
        min_instances_slice: int = 100,
        scope=None,
        clustering_method: Union[str, Callable] = "kmeans",
        partitioning_method: Union[str, Callable] = "rdc",
        clustering_args: Optional[dict[str, Any]] = None,
        partitioning_args: Optional[dict[str, Any]] = None,
        check_support: bool = True,
        full_data: torch.tensor = None,
) -> Module:
    """LearnSPN structure and parameter learner for the ``base`` backend.

    LearnSPN algorithm as described in (Gens & Domingos, 2013): "Learning the Structure of Sum-Product Networks".

    Args:
        data:
            torchwo-dimensional NumPy array containing the input data.
            Each row corresponds to a sample.
        feature_ctx:
            ``FeatureContext`` instance specifying the domains of the scopes.
            Scope query RVs must match the data features.
            Defaults to None, in which case a feature context is initialized from ``data``.
        min_features_slice:
            Integer value specifying the minimum number of features required to partition.
            Defaults to 2.
        min_instances_slice:
            Integer value specifying the minimum number of instances required to cluster.
            Defaults to 100.
        clustering_method:
            String or callable specifying the clustering method to be used.
            If 'kmeans' k-Means clustering is used.
            If a callable, it is expected to accept ``data`` and return a one-dimensional NumPy array of integer values indicating the clusters the corresponding samples are assigned to.
        partitioning_method:
            String or callable specifying the partitioning method to be used.
            If 'rdc' randomized dependence coefficients (RDCs) are used to determine independencies.
            If a callable, it is expected to accept ``data`` and return a one-dimensional NumPy array with the same number of features as in ``data`` of integer values indicating the partitions the corresponding features are assigned to.
        clustering_args:
            Optional dictionary mapping keyword arguments to objects.
            Passed to ``clustering_method`` each time it is called.
        partitioning_args:
            Optional dictionary mapping keyword arguments to objects.
            Passed to ``partitioning_method`` each time it is called.
        check_support:
            Boolean value indicating whether or not if the data is in the support of the leaf distributions.
            Defaults to True.

    Returns:
        A node representing the learned SPN.

    Raises:
        ValueError: Invalid arguments.
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

    if len(scope.query) != data.shape[1]:
        raise ValueError(f"Number of query variables in 'scope' does not match number of features in data.")

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

    def create_partitioned_mv_leaf(scope: Scope,data: torch.Tensor):
        # create leaf layer from given scope and data
        leaves = []
        s = set(scope.query)
        for leaf_module in leaf_modules:
            leaf_scope = set(leaf_module.scope.query)
            scope_inter = s.intersection(leaf_scope)
            if len(scope_inter) > 0:
                leaf_layer = leaf_module.__class__(scope=Scope(sorted(scope_inter)),
                                                   out_channels=leaf_module.out_channels)
                # estimate leaf node parameters from data
                maximum_likelihood_estimation(leaf_layer, data, check_support=check_support, preprocess_data=False)

                leaves.append(leaf_layer)

        if len(scope.query) > 1:
            if len(leaves) == 1:
                return Product(inputs=leaves[0])
            else:
                return Product(inputs=Cat(leaves, dim=1))
        else:
            return leaves[0]

    # features does not need to be split any further
    if len(scope.query) < min_features_slice:
        return create_partitioned_mv_leaf(scope, data)

    else:

        # select correct data
        if not data.shape[0] == 1:
            partition_ids = partitioning_method(data)  # uc

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
                    data[:, partition[0]],
                    leaf_modules=leaf_modules,
                    scope=Scope([scope.query[rv] for rv in partition[0]]),
                    out_channels=out_channels,
                    clustering_method=clustering_method,
                    partitioning_method=partitioning_method,
                    min_features_slice=min_features_slice,
                    min_instances_slice=min_instances_slice,
                )
                product_inputs.append(sub_structure)

            return Product(inputs=Cat(product_inputs, dim=1))


        else:
            # if not enough instances to cluster, create leaf layer (can be set to prevent overfitting too much or to reduce network size)
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
