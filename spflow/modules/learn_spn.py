"""Contains the LearnSPN structure and parameter learner for SPFlow in the ``base`` backend.
"""
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
from spflow.utils.kmeans_new import KMeans as KMeans_torch

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.utils.connected_components import connected_components
from spflow.utils.randomized_dependency_coefficients import (
    randomized_dependency_coefficients,
)
from spflow.modules.leaf.leaf_module import LeafModule
from networkx import connected_components as ccnp, from_numpy_array



def partition_by_rdc(
    data: torch.torch.Tensor,
    threshold: float = 0.3,
    preprocessing: Optional[Callable] = None,
) -> torch.torch.Tensor:
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

    #print("partitiong start")

    # get pairwise rdc values
    rdcs = randomized_dependency_coefficients(partitioning_data)

    # create adjacency matrix of features from thresholded rdcs
    adj_mat = torch.tensor((rdcs >= threshold), dtype=torch.int)

    partition_ids = torch.zeros(data.shape[1], dtype=torch.int)

    #for i, cc in enumerate(connected_components(adj_mat)):
    #    partition_ids[list(cc)] = i
    for i, c in enumerate((ccnp(from_numpy_array(np.array(adj_mat))))):
        partition_ids[list(c)] = i

    #print("partitiong end")

    return partition_ids


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

    #print("kmeans start")
    #_, data_labels = kmeans(clustering_data, n_clusters=n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
    data_labels = kmeans.fit_predict(clustering_data)
    #
    print("kmeans done")


    return data_labels


def learn_spn(
    data: torch.Tensor,
    leaf_modules: Union[list[LeafModule], LeafModule],
    #feature_ctx: Optional[FeatureContext] = None,
    min_features_slice: int = 2,
    min_instances_slice: int = 100,
    fit_params: bool = True,
    scope = None,
    clustering_method: Union[str, Callable] = "kmeans",
    partitioning_method: Union[str, Callable] = "rdc",
    clustering_args: Optional[dict[str, Any]] = None,
    partitioning_args: Optional[dict[str, Any]] = None,
    check_support: bool = True,
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
        fit_params:
            Boolean value determining whether or not to estimate the parameters of the nodes.
            If set to False, only the structure is learned. Can not be enabled if a conditional SPN structure is to be learned.
            Defaults to True.
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
    # initialize feature context
    #if feature_ctx is None:
    #    feature_ctx = FeatureContext(Scope(list(range(data.shape[1]))))
    if scope is None:
        if isinstance(leaf_modules, list):
            if len(leaf_modules) > 1:
                assert Scope.all_pairwise_disjoint([module.scope for module in leaf_modules]), "Leaf modules must have disjoint scopes."
                scope = leaf_modules[0].scope
                for leaf in leaf_modules[1:]:
                    scope = scope.join(leaf.scope)
            else:
                scope = leaf_modules[0].scope
        else:
            scope = leaf_modules.scope
            leaf_modules = [leaf_modules]


    #scope = feature_ctx.scope

    if len(scope.query) != data.shape[1]:
        raise ValueError(f"Number of query variables in 'scope' does not match number of features in data.")
    if scope.is_conditional() and fit_params:
        raise ValueError("Option 'fit_params' is set to True, but is incompatible with a conditional scope.")

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

    # helper functions
    def fit_leaf(leaf: Module, data: torch.Tensor, scope: Scope):
        # create empty data set with data at correct leaf scope indices
        leaf_data = torch.zeros((data.shape[0], max(scope.query) + 1), dtype=data.dtype, device=data.device)
        leaf_data[:, scope.query] = data[:, [scope.query.index(rv) for rv in scope.query]]

        # estimate leaf node parameters from data
        maximum_likelihood_estimation(leaf, leaf_data, check_support=check_support)
    """
    def create_uv_leaf(scope: Scope, data: torch.Tensor, fit_params: bool = True):
        # create leaf node
        signature = feature_ctx.select(scope.query)
        leaf = AutoLeaf([signature])

        if fit_params:
            fit_leaf(leaf, data, scope)

        return leaf
    """
    def create_partitioned_mv_leaf(scope: Scope, data: torch.Tensor, fit_params: bool = True):
        # combine univariate leafs via product node
        leaves = []
        #for rv in scope.query:
            # create leaf node
            #signature = feature_ctx.select([rv])
            #leaf = AutoLeaf([signature])
            #leaves.append(leaf)
        s = set(scope.query)
        for leaf_module in leaf_modules:
            leaf_scope = set(leaf_module.scope.query)
            scope_inter = s.intersection(leaf_scope)
            if len(scope_inter) > 0:
                leaf_layer = leaf_module.__class__(scope=Scope(sorted(scope_inter)), out_channels=leaf_module.out_channels)
                if fit_params:
                    fit_leaf(leaf_layer, data, scope)
                leaves.append(leaf_layer)


        if len(scope.query) > 1:
            return Product(inputs= Cat(leaves, dim=1)) #ToDo: How to input the leaves?
        else:
            return leaves[0]

    # features does not need to be split any further
    if len(scope.query) < min_features_slice:
        # multivariate scope
        # torchoDO: unecessary distinction between univariate and multivariate scope ??
        """
        if len(scope.query) > 1:
            return create_partitioned_mv_leaf(scope, data, fit_params)
        # univariate scope
        else:
            return create_uv_leaf(scope, data, fit_params)
        """
        return create_partitioned_mv_leaf(scope, data, fit_params)
    else:
        # select correct data
        partition_ids = partitioning_method(data)

        # compute partitions of rvs from partition id labels
        partitions = []

        #for partition_id in torch.sort(torch.unique(partition_ids), axis=-1):
        #    partitions.append(torch.where(partition_ids == partition_id)[0])

        for partition_id in torch.sort(torch.unique(partition_ids), axis=-1)[0]:
            partitions.append(torch.where(partition_ids == partition_id))

        # multiple partition (i.e., data can be partitioned)
        if len(partitions) > 1:
            return Product(
                inputs=Cat([
                    # compute child trees recursively
                    learn_spn(
                        data[:, partition[0]],
                        leaf_modules=leaf_modules,
                        scope=Scope([scope.query[rv] for rv in partition[0]]),
                        #feature_ctx=feature_ctx.select([scope.query[rv] for rv in partition]),
                        clustering_method=clustering_method,
                        partitioning_method=partitioning_method,
                        fit_params=fit_params,
                        min_features_slice=min_features_slice,
                        min_instances_slice=min_instances_slice,
                    )
                    for partition in partitions
                ], dim=1)
            )
        else:
            # if not enough instances to cluster, create univariate leaf nodes (can be set to prevent overfitting too much or to reduce network size)
            if data.shape[0] < min_instances_slice:
                return create_partitioned_mv_leaf(scope, data, fit_params)
            # cluster data
            else:
                labels = clustering_method(data)

                # non-conditional clusters
                if not scope.is_conditional():
                    weights = (
                        torch.Tensor([torch.sum(labels == cluster_id) / data.shape[0] for cluster_id in torch.unique(labels)])
                        if fit_params
                        else None
                    )

                    return Sum(
                        inputs=Cat([
                            learn_spn(
                                data[labels == cluster_id, :],
                                leaf_modules=leaf_modules,
                                #feature_ctx=feature_ctx,
                                scope=scope,
                                clustering_method=clustering_method,
                                partitioning_method=partitioning_method,
                                fit_params=fit_params,
                                min_features_slice=min_features_slice,
                                min_instances_slice=min_instances_slice,
                            )
                            for cluster_id in torch.unique(labels)
                        ], dim=2),
                        weights=None,#weights,
                        out_channels=2
                    )
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
                                fit_params=fit_params,
                                min_features_slice=min_features_slice,
                                min_instances_slice=min_instances_slice,
                            )
                            for cluster_id in torch.unique(labels)
                        ],
                    )
                    """
