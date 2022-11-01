# -*- coding: utf-8 -*-
"""Contains the LearnSPN structure and parameter learner for SPFlow in the ``base`` backend.
"""
import numpy as np
from functools import partial
from typing import Callable, Union, Optional, Dict, Any
from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.utils.randomized_dependency_coefficients import (
    randomized_dependency_coefficients,
)
from spflow.base.learning.nodes.leaves.parametric.gaussian import (
    maximum_likelihood_estimation,
)
from spflow.base.utils.connected_components import connected_components
from spflow.base.structure.autoleaf import AutoLeaf
from spflow.base.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.base.structure.nodes.cond_node import SPNCondSumNode
from spflow.base.structure.module import Module
from sklearn.cluster import KMeans


def partition_by_rdc(
    data: np.ndarray,
    threshold: float = 0.3,
    preprocessing: Optional[Callable] = None,
) -> np.ndarray:
    """Performs partitioning usig randomized dependence coefficients (RDCs) to be used with the LearnSPN algorithm in the ``base`` backend.

    Args:
        data:
            Two-dimensional NumPy array containing the input data.
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

    # get pairwise rdc values
    rdcs = randomized_dependency_coefficients(partitioning_data)

    # create adjacency matrix of features from thresholded rdcs
    adj_mat = (rdcs >= threshold).astype(int)

    partition_ids = np.zeros(data.shape[1])

    for i, cc in enumerate(connected_components(adj_mat)):
        partition_ids[list(cc)] = i

    return partition_ids


def cluster_by_kmeans(
    data: np.ndarray,
    n_clusters: int = 2,
    preprocessing: Optional[Callable] = None,
) -> np.ndarray:
    """Performs clustering usig k-Means to be used with the LearnSPN algorithm in the ``base`` backend.

    Args:
        data:
            Two-dimensional NumPy array containing the input data.
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

    # compute k-Means clusters
    data_labels = KMeans(n_clusters=n_clusters).fit_predict(clustering_data)

    return data_labels


def learn_spn(
    data: np.ndarray,
    feature_ctx: Optional[FeatureContext] = None,
    min_features_slice: int = 2,
    min_instances_slice: int = 100,
    fit_params: bool = True,
    clustering_method: Union[str, Callable] = "kmeans",
    partitioning_method: Union[str, Callable] = "rdc",
    clustering_args: Optional[Dict[str, Any]] = None,
    partitioning_args: Optional[Dict[str, Any]] = None,
    check_support: bool = True,
) -> Module:
    """LearnSPN structure and parameter learner for the ``base`` backend.

    LearnSPN algorithm as described in (Gens & Domingos, 2013): "Learning the Structure of Sum-Product Networks".

    Args:
        data:
            Two-dimensional NumPy array containing the input data.
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
    if feature_ctx is None:
        feature_ctx = feature_ctx(Scope(list(range(data.shape[1]))))

    scope = feature_ctx.scope

    if len(scope.query) != data.shape[1]:
        raise ValueError(
            f"Number of query variables in 'scope' does not match number of features in data."
        )
    if scope.is_conditional() and fit_params:
        raise ValueError(
            "Option 'fit_params' is set to True, but is incompatible with a conditional scope."
        )

    # available off-the-shelf clustering methods provided by SPFlow
    if isinstance(clustering_method, str):
        # Randomized Dependence Coefficients (RDCs)
        if clustering_method == "kmeans":
            clustering_method = cluster_by_kmeans
        else:
            raise ValueError(
                f"Value '{clustering_method}' for partitioning method is invalid."
            )

    # available off-the-shelf partitioning methods provided by SPFlow
    if isinstance(partitioning_method, str):
        # Randomized Dependence Coefficients (RDCs)
        if partitioning_method == "rdc":
            partitioning_method = partition_by_rdc
        else:
            raise ValueError(
                f"Value '{partitioning_method}' for partitioning method is invalid."
            )

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
    def create_uv_leaf(scope: Scope, data: np.ndarray, fit_params: bool = True):
        # create leaf node
        signature = feature_ctx.select(scope.query)
        leaf = AutoLeaf([signature])

        if fit_params:
            # estimate leaf node parameters from data
            maximum_likelihood_estimation(
                leaf, data, check_support=check_support
            )

        return leaf

    def create_partitioned_mv_leaf(
        scope: Scope, data: np.ndarray, fit_params: bool = True
    ):
        # combine univariate leafs via product node
        leaves = []

        for rv in scope.query:
            # create leaf node
            signature = feature_ctx.select([rv])
            leaf = AutoLeaf([signature])
            leaves.append(leaf)

            if fit_params:
                # estimate leaf node parameters from data
                maximum_likelihood_estimation(
                    leaf, data[:, [rv]], check_support=check_support
                )

        return SPNProductNode(children=leaves)

    # features does not need to be split any further
    if len(scope.query) < min_features_slice:

        # multivariate scope
        if len(scope.query) > 1:
            return create_partitioned_mv_leaf(scope, data, fit_params)
        # univariate scope
        else:
            return create_uv_leaf(scope, data, fit_params)
    else:
        # select correct data
        partition_ids = partitioning_method(data)

        # compute partitions of rvs from partition id labels
        partitions = []

        for partition_id in np.sort(np.unique(partition_ids)):
            partitions.append(np.where(partition_ids == partition_id)[0])

        # multiple partition (i.e., data can be partitioned)
        if len(partitions) > 1:
            return SPNProductNode(
                children=[
                    # compute child trees recursively
                    learn_spn(
                        data[:, partition],
                        feature_ctx=feature_ctx.select(
                            [scope.query[rv] for rv in partition]
                        ),
                        clustering_method=clustering_method,
                        partitioning_method=partitioning_method,
                        fit_params=fit_params,
                        min_features_slice=min_features_slice,
                        min_instances_slice=min_instances_slice,
                    )
                    for partition in partitions
                ]
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
                        [
                            (labels == cluster_id).sum() / data.shape[0]
                            for cluster_id in np.unique(labels)
                        ]
                        if fit_params
                        else None
                    )

                    return SPNSumNode(
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
                            for cluster_id in np.unique(labels)
                        ],
                        weights=weights,
                    )
                # conditional clusters
                else:
                    return SPNCondSumNode(
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
                            for cluster_id in np.unique(labels)
                        ],
                    )
