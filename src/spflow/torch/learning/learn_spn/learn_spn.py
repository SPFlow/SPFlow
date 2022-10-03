"""
Created on September 26, 2022

@authors: Philipp Deibert
"""
import torch
from functools import partial
from typing import Callable, Union, Optional, Dict, Any
from spflow.meta.scope.scope import Scope
from spflow.torch.learning.nodes.leaves.parametric.gaussian import maximum_likelihood_estimation
from spflow.torch.utils.randomized_dependency_coefficients import randomized_dependency_coefficients
from spflow.torch.utils.connected_components import connected_components
from spflow.torch.utils.kmeans import kmeans
from spflow.torch.structure.nodes.leaves.parametric.gaussian import Gaussian
from spflow.torch.structure.nodes.node import SPNSumNode, SPNProductNode
from spflow.torch.structure.module import Module


def partition_by_rdc(data: torch.Tensor, threshold: float=0.3, preprocessing: Optional[Callable]=None) -> torch.Tensor:

    # perform optional pre-processing of data
    if preprocessing is not None:
        partitioning_data = preprocessing(data)
    else:
        partitioning_data = data

    # get pairwise rdc values
    rdcs = randomized_dependency_coefficients(partitioning_data)

    # create adjacency matrix of features from thresholded rdcs
    adj_mat = (rdcs >= threshold).type(torch.long)

    partition_ids = torch.zeros(data.shape[1])

    for i, cc in enumerate(connected_components(adj_mat)):
        partition_ids[list(cc)] = i

    return partition_ids


def cluster_by_kmeans(data: torch.Tensor, n_clusters: int=2, preprocessing: Optional[Callable]=None) -> torch.Tensor:

    # perform optional pre-processing of data
    if preprocessing is not None:
        clustering_data = preprocessing(data)
    else:
        clustering_data = data
    
    # compute k-Means clusters
    _, data_labels = kmeans(clustering_data, n_clusters=n_clusters)

    return data_labels


def learn_spn(data, scope: Optional[Scope]=None, min_features_slice: int=2, min_instances_slice: int=100, fit_leaves: bool=True, clustering_method: Union[str, Callable]="kmeans", partitioning_method: Union[str, Callable]="rdc", clustering_args: Optional[Dict[str, Any]]=None, partitioning_args: Optional[Dict[str, Any]]=None) -> Module:

    # initialize scope
    if scope is None:
        scope = Scope(list(range(data.shape[1])))
    else:
        if len(scope.evidence) != 0:
            raise ValueError("Scope specified for 'learn_spn' may not contain evidence variables.")
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
        raise ValueError(f"Value for 'min_instances_slice' must be an integer greater than 1, but was: {min_instances_slice}.")
    if not isinstance(min_features_slice, int) or min_features_slice < 2:
        raise ValueError(f"Value for 'min_features_slice' must be an integer greater than 1, but was: {min_features_slice}.")

    # helper functions
    def create_uv_leaf(scope: Scope, data: torch.Tensor, fit_leaves: bool=True):
        # create leaf node
        leaf = Gaussian(scope=scope) # TODO: infer correct leaf node

        if fit_leaves:
            # estimate leaf node parameters from data
            maximum_likelihood_estimation(leaf, data)
        
        return leaf

    def create_partitioned_mv_leaf(scope: Scope, data: torch.Tensor, fit_leaves: bool=True):
        # combine univariate leafs via product node
        leaves = []

        for rv in scope.query:
            # create leaf node
            leaf = Gaussian(scope=Scope([rv]))
            leaves.append(leaf)

            if fit_leaves:
                # estimate leaf node parameters from data
                maximum_likelihood_estimation(leaf, data[:, [rv]])

        return SPNProductNode(children=leaves)

    # features does not need to be split any further
    if len(scope.query) < min_features_slice:

        # multivariate scope
        if len(scope.query) > 1:
            return create_partitioned_mv_leaf(scope, data, fit_leaves)
        # univariate scope
        else:
            return create_uv_leaf(scope, data, fit_leaves)
    else:
        # select correct data
        partition_ids = partitioning_method(data)

        # compute partitions of rvs from partition id labels
        partitions = []

        for partition_id in torch.sort(torch.unique(partition_ids)).values:
            partitions.append( torch.where(partition_ids == partition_id)[0] )

        # multiple partition (i.e., data can be partitioned)
        if len(partitions) > 1:
            return SPNProductNode(
                children=[
                    # compute child trees recursively
                    learn_spn(data[:, partition],
                            scope=Scope([scope.query[rv] for rv in partition]),
                            clustering_method=clustering_method,
                            partitioning_method=partitioning_method,
                            fit_leaves=fit_leaves,
                            min_features_slice=min_features_slice,
                            min_instances_slice=min_instances_slice
                        ) for partition in partitions
                ]
            )
        else:
            # if not enough instances to cluster, create univariate leaf nodes (can be set to prevent overfitting too much or to reduce network size)
            if data.shape[0] < min_instances_slice:
                return create_partitioned_mv_leaf(scope, data, fit_leaves)
            # cluster data
            else:
                labels = clustering_method(data)

                return SPNSumNode(
                    children=[
                        learn_spn(data[labels == cluster_id, :],
                                scope=scope,
                                clustering_method=clustering_method,
                                partitioning_method=partitioning_method,
                                fit_leaves=fit_leaves,
                                min_features_slice=min_features_slice,
                                min_instances_slice=min_instances_slice
                            ) for cluster_id in torch.unique(labels)
                    ],
                    weights=[(labels == cluster_id).sum()/data.shape[0] for cluster_id in torch.unique(labels)]
                )