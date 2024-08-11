"""Algorithm to compute k-Means clusters from data.

Typical usage example:

    labels = kmeans(data, n_clusters)
"""
from typing import Tuple, Union

import torch


def kmeans(
    data: torch.Tensor,
    n_clusters: int,
    centroids: Union[str, torch.Tensor] = "kmeans++",
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes k-Means clustering on input data.

    Performs k-Means clustering on input data and returns the resulting cluster labels for all input samples as well as the cluster centroids.
    Implementation follows the efficient algorithm described in (Hamerly, 2010): "Making k-means even faster".

    Args:
        data:
            Two-dimensional PyTorch tensor containing the data to be clustered. Each row is regarded as a (possibly multivariate) sample.
        n_clusters:
            Integer specifying the desired number of clusters to be used. Must be at least 1.
        centroids:
            String or PyTorch tensor specifying which initial centroids should be used.
            In case of 'kmeans++', the first centroid is chosen randomly from data and each successive centroid is selected to be furthest away from any existing centroid.
            In case of 'random', centroids are randomly chosen from data samples.
            If a PyTorch tensor is specified, it is expected to be a two-dimensional where each row is a different centroid. Number of centroids must match number of specified clusters.
            Defaults to 'kmeans++'.
        max_iter:
            Maximum number of iterations. Defaults to 300.
        tol:
            Tolerance to check for convergence of algorithm. Defaults to 10^(-4).

    Returns:
        Tuple of PyTorch tensors. The first tensor represents the cluster labels for the input samples. The second tensor contains the final cluster centroids.

    Raises:
        ValueError: Invalid argument values.
    """
    if n_clusters < 1:
        raise ValueError(
            f"k-Means clustering requires at least one target cluster, but {n_clusters} were specified."
        )

    if n_clusters > data.shape[0]:
        raise ValueError(
            f"k-Means clusterin requires number of desired clusters to be less or equal to the number of data points."
        )

    if isinstance(centroids, str):
        if centroids == "kmeans++":
            # select first centroid randomly from data
            centroids = data[torch.randint(0, n_clusters, (1,))].reshape(1, -1)

            # select each additional centroid to be farthest away from any other selected centroid
            for i in range(1, n_clusters):
                # smallest distances from each data point to any centroid
                d = torch.min(torch.cdist(data, centroids, p=2), dim=1).values

                # add furthest point to centroids
                centroids = torch.vstack([centroids, data[d.argmax()]])
        elif centroids == "random":
            # pick random data points as starting centroids
            centroids = data[torch.randperm(len(data))[:n_clusters]]
        else:
            raise ValueError(f"Unknown initialization strategy {centroids} for k-Means clustering.")

    if centroids.ndim == 1:
        # make sure centroids tensor is 2-dimensional (n_centroids, n_features)
        centroids = centroids.reshape(-1, 1)

    if centroids.shape[0] != n_clusters:
        raise ValueError(
            f"Number of specified centroids {centroids.shape[0]} does not match number of desired clusters {n_clusters}."
        )

    if centroids.shape[1] != data.shape[1]:
        raise ValueError(
            f"Number of centroid features {centroids.shape[1]} does not match number of data features {data.shape[1]}."
        )

    n_features = data.shape[1]

    # zero-center data for more precision in distance computations
    data_mean = data.mean(dim=0)
    data = data - data_mean

    # Implementation follows (Hamerly, 2010): https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf

    # ----- initialization ------

    # compute euclidean distances of each point to different centroids
    d = torch.cdist(data, centroids, p=2)
    d_sorted_ids = torch.argsort(d, dim=1)

    # ids of closest centroids (a in paper)
    assigned_labels = d_sorted_ids[:, 0]

    # distances to closest centroids (u in paper)
    upper_bounds = d.gather(1, assigned_labels.unsqueeze(1)).squeeze(1)
    # distances to second closest centroids (l in paper)
    lower_bounds = d.gather(1, d_sorted_ids[:, 1].unsqueeze(1)).squeeze(1)

    # points per cluster (q in paper)
    points_per_cluster = torch.zeros(n_clusters)
    # sums of points in cluster (c_prime in paper)
    cluster_sums = torch.zeros(n_clusters, n_features)

    for cluster_id in range(n_clusters):
        points_per_cluster[cluster_id] = (assigned_labels == cluster_id).sum()
        cluster_sums[cluster_id] = data[assigned_labels == cluster_id].sum(dim=0)

    # ----- iterations -----

    # keep track of previous centroids for convergence testing
    update_sizes = torch.ones(n_clusters)

    while torch.any(update_sizes > tol) or max_iter > 0:
        # closest euclidean distance from one cluster to another
        cc = torch.sort(torch.cdist(centroids, centroids, p=2)).values[:, 1]

        m = torch.max(torch.vstack([cc[assigned_labels] / 2.0, lower_bounds]), dim=0).values

        # select points for which upper bound is larger than m
        bound_mask = upper_bounds > m

        # update upper bounds (set to euclidean distance to assigned cluster)
        upper_bounds[bound_mask] = torch.sqrt(
            torch.pow(data[bound_mask] - centroids[assigned_labels[bound_mask]], 2).sum(dim=1)
        )

        # select points for which upper bound is still larger than m
        bound_mask &= upper_bounds > m

        prev_assigned_labels = torch.clone(assigned_labels)  # a_prime in paper

        # compute euclidean distances of selected point to different centroids
        d = torch.cdist(data[bound_mask], centroids, p=2)
        d_sorted_ids = torch.argsort(d, dim=1)

        # update cluster assignements
        assigned_labels[bound_mask] = d_sorted_ids[:, 0]

        # update upper bounds
        upper_bounds[bound_mask] = d.gather(1, assigned_labels[bound_mask].unsqueeze(1)).squeeze(1)
        # update lower bounds
        lower_bounds[bound_mask] = d.gather(1, d_sorted_ids[:, 1].unsqueeze(1)).squeeze(1)

        # select points for which assigned cluster changed
        bound_mask &= prev_assigned_labels != assigned_labels

        # TODO: can potentially be done more efficiently
        for cluster_id in range(n_clusters):
            excluded = bound_mask & (prev_assigned_labels == cluster_id)
            included = bound_mask & (assigned_labels == cluster_id)

            # update number of points in cluster
            points_per_cluster[cluster_id] += included.sum() - excluded.sum()

            # update sum of cluster points
            cluster_sums[cluster_id] += data[included, :].sum(dim=0) - data[excluded, :].sum(dim=0)

        # update centroids
        centroids_prev = torch.clone(centroids)

        # if any clusters
        centroids = cluster_sums / points_per_cluster.unsqueeze(1)

        # compute eucliden change for each cluster
        update_sizes = torch.sqrt(torch.pow(centroids_prev - centroids, 2).sum(dim=1))

        # update bounds
        ordered_cluster_changes = torch.argsort(update_sizes, descending=True)
        most_updated = ordered_cluster_changes[0]
        second_most_updated = ordered_cluster_changes[1]

        upper_bounds += update_sizes[assigned_labels]
        lower_bounds[most_updated] -= update_sizes[most_updated]
        lower_bounds[second_most_updated] -= update_sizes[second_most_updated]

        # decrement max iteration counter
        if max_iter != -1:
            max_iter -= 1

    # add data mean to centroids
    centroids += data_mean

    return centroids, assigned_labels