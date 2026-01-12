"""Prometheus structure learning algorithm for SPNs.

Implements the Prometheus algorithm from:
  Priyank Jaini, Amur Ghose, Pascal Poupart (2018)
  "Prometheus: Directly Learning Acyclic Directed Graph Structures for Sum-Product Networks"

Prometheus learns a DAG-structured SPN by:
- clustering instances (mixture modeling),
- generating multiple variable decompositions per cluster via maximum spanning tree (MST) partitioning,
- sharing repeated sub-scopes across decompositions (subtree sharing / DAG),
- recursively applying the procedure to each sub-scope.

    This implementation supports both the full O(d^2) affinity matrix and the scalable sampling-based
    approximation described in the paper (via `affinity_mode="sampled"`).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from itertools import combinations
import math
from typing import Any

import numpy as np
import networkx as nx
import torch
from fast_pytorch_kmeans import KMeans
from networkx import connected_components, from_numpy_array, maximum_spanning_tree

from spflow.exceptions import InvalidParameterError, InvalidTypeError, UnsupportedOperationError
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.rdc import rdc


def _adapt_product_inputs(inputs: list[Module]) -> list[Module]:
    """Ensure all product inputs have the same number of channels.

    Product nodes concatenate children along feature dimension; this requires all children
    to have the same number of channels. When they differ, wrap smaller-channel children
    in a Sum module to match the maximum channels among siblings (as done in learn_spn).
    """
    target_channels = max(m.out_shape.channels for m in inputs)
    output_modules: list[Module] = []
    for m in inputs:
        if m.out_shape.channels < target_channels:
            output_modules.append(Sum(inputs=m, out_channels=target_channels))
        else:
            output_modules.append(m)
    return output_modules


def _cluster_by_kmeans(
    data: torch.Tensor,
    n_clusters: int = 2,
    preprocessing: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Cluster rows using k-means.

    Args:
        data: 2D tensor (batch, features).
        n_clusters: Number of clusters.
        preprocessing: Optional preprocessing applied before clustering.

    Returns:
        1D tensor of integer cluster ids of length batch.
    """
    if preprocessing is not None:
        clustering_data = preprocessing(data)
    else:
        clustering_data = data

    kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", verbose=0)
    return kmeans.fit_predict(clustering_data)


def _affinity_corr(data: torch.Tensor) -> torch.Tensor:
    """Compute absolute Pearson correlation affinity between columns.

    Args:
        data: 2D tensor (batch, d) without NaNs.

    Returns:
        2D tensor (d, d) with values in [0, 1].
    """
    if data.dim() != 2:
        raise InvalidParameterError(f"Expected 2D data for affinity, got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise UnsupportedOperationError("Prometheus affinity does not support NaNs; impute first.")

    d = data.shape[1]
    if d == 1:
        return torch.ones((1, 1), device=data.device, dtype=data.dtype)

    x = data - data.mean(dim=0, keepdim=True)
    denom = torch.sqrt(torch.sum(x * x, dim=0, keepdim=True)).clamp_min(1e-12)
    x = x / denom
    corr = (x.t() @ x).clamp(min=-1.0, max=1.0).abs()
    corr.fill_diagonal_(1.0)
    return corr


def _affinity_rdc(data: torch.Tensor) -> torch.Tensor:
    """Compute RDC affinity between all column pairs.

    Args:
        data: 2D tensor (batch, d) without NaNs.

    Returns:
        2D tensor (d, d) with values in [0, 1].
    """
    if data.dim() != 2:
        raise InvalidParameterError(f"Expected 2D data for affinity, got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise UnsupportedOperationError("Prometheus affinity does not support NaNs; impute first.")

    d = data.shape[1]
    aff = torch.eye(d, device=data.device, dtype=data.dtype)
    for i, j in combinations(range(d), 2):
        value = rdc(data[:, i], data[:, j])
        aff[i, j] = value
        aff[j, i] = value
    return aff


def _mst_partitions_from_affinity(affinity: torch.Tensor) -> list[list[list[int]]]:
    """Generate component partitions by iteratively severing weakest MST edges.

    Args:
        affinity: 2D tensor (d, d) representing an undirected complete weighted graph.

    Returns:
        A list of partitions; each partition is a list of components; each component is
        a list of local variable indices in [0, d).
        The first partition corresponds to severing 1 edge (2 components) and the last
        corresponds to severing d-1 edges (d components).
    """
    if affinity.dim() != 2 or affinity.shape[0] != affinity.shape[1]:
        raise InvalidParameterError(f"Affinity must be square, got shape {tuple(affinity.shape)}.")

    d = affinity.shape[0]
    if d <= 1:
        return []

    graph = from_numpy_array(np.asarray(affinity.detach().cpu().numpy()))
    tree = maximum_spanning_tree(graph, weight="weight")

    edges_sorted = sorted(tree.edges(data=True), key=lambda e: float(e[2].get("weight", 0.0)))
    partitions: list[list[list[int]]] = []

    for u, v, _data in edges_sorted:
        if tree.has_edge(u, v):
            tree.remove_edge(u, v)
        comps = [sorted(list(c)) for c in connected_components(tree)]
        partitions.append(comps)

    return partitions


def _default_samples_per_var(num_features: int) -> int:
    """Compute the default number of sampled neighbors per variable."""
    if num_features <= 1:
        return 0
    return max(1, min(num_features - 1, int(math.log2(num_features))))


def _sampled_edges_corr(
    data: torch.Tensor,
    samples_per_var: int,
    seed: int | None,
) -> list[tuple[int, int, float]]:
    """Compute sampled absolute Pearson correlations as weighted edges.

    Args:
        data: 2D tensor (batch, d) without NaNs.
        samples_per_var: Number of neighbors sampled per variable.
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        List of weighted undirected edges (u, v, weight), with u < v.
    """
    if data.dim() != 2:
        raise InvalidParameterError(f"Expected 2D data for affinity, got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise UnsupportedOperationError("Prometheus affinity does not support NaNs; impute first.")

    d = data.shape[1]
    if d <= 1 or samples_per_var <= 0:
        return []

    x = data - data.mean(dim=0, keepdim=True)
    denom = torch.sqrt(torch.sum(x * x, dim=0, keepdim=True)).clamp_min(1e-12)
    x = x / denom

    rng = np.random.default_rng(seed)
    edges: dict[tuple[int, int], float] = {}
    k = min(samples_per_var, d - 1)
    for i in range(d):
        choices = rng.choice(d - 1, size=k, replace=False)
        for c in choices:
            j = int(c) if c < i else int(c) + 1
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges:
                continue
            weight = float(torch.dot(x[:, a], x[:, b]).abs().item())
            edges[(a, b)] = weight
    return [(u, v, w) for (u, v), w in edges.items()]


def _sampled_edges_rdc(
    data: torch.Tensor,
    samples_per_var: int,
    seed: int | None,
) -> list[tuple[int, int, float]]:
    """Compute sampled RDC affinities as weighted edges.

    Args:
        data: 2D tensor (batch, d) without NaNs.
        samples_per_var: Number of neighbors sampled per variable.
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        List of weighted undirected edges (u, v, weight), with u < v.
    """
    if data.dim() != 2:
        raise InvalidParameterError(f"Expected 2D data for affinity, got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise UnsupportedOperationError("Prometheus affinity does not support NaNs; impute first.")

    d = data.shape[1]
    if d <= 1 or samples_per_var <= 0:
        return []

    rng = np.random.default_rng(seed)
    edges: dict[tuple[int, int], float] = {}
    k = min(samples_per_var, d - 1)
    for i in range(d):
        choices = rng.choice(d - 1, size=k, replace=False)
        for c in choices:
            j = int(c) if c < i else int(c) + 1
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges:
                continue
            value = rdc(data[:, a], data[:, b])
            edges[(a, b)] = float(value)
    return [(u, v, w) for (u, v), w in edges.items()]


def _mst_partitions_from_sampled_edges(
    num_features: int,
    edges: list[tuple[int, int, float]],
) -> list[list[list[int]]]:
    """Generate partitions using a max spanning tree built from sampled edges.

    Missing edges are treated as weight 0; if the sampled graph is disconnected,
    components are connected with zero-weight edges in a deterministic chain.
    """
    if num_features <= 1:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(num_features))
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    tree = maximum_spanning_tree(graph, weight="weight")
    components = [sorted(list(c)) for c in connected_components(tree)]
    if len(components) > 1:
        reps = [c[0] for c in sorted(components, key=lambda c: c[0])]
        for u, v in zip(reps, reps[1:]):
            tree.add_edge(u, v, weight=0.0)

    edges_sorted = sorted(tree.edges(data=True), key=lambda e: float(e[2].get("weight", 0.0)))
    partitions: list[list[list[int]]] = []
    for u, v, _data in edges_sorted:
        if tree.has_edge(u, v):
            tree.remove_edge(u, v)
        comps = [sorted(list(c)) for c in connected_components(tree)]
        partitions.append(comps)
    return partitions


def _infer_scope_from_leaf_modules(leaf_modules: list[LeafModule] | LeafModule) -> Scope:
    """Infer the global scope from leaf module templates.

    Args:
        leaf_modules: Single leaf module or list of leaf modules.

    Returns:
        Scope covering the union of provided leaf scopes.
    """
    if isinstance(leaf_modules, list):
        if not leaf_modules:
            raise InvalidParameterError("'leaf_modules' must not be empty.")
        scope = leaf_modules[0].scope
        for leaf in leaf_modules[1:]:
            scope = scope.join(leaf.scope)
        return scope
    return leaf_modules.scope


def _create_partitioned_mv_leaf(
    scope: Scope,
    data: torch.Tensor,
    leaf_modules: list[LeafModule],
) -> Module:
    """Create leaf distribution(s) for a (possibly multivariate) scope.

    SPFlow leaf modules are provided as templates; this function instantiates new
    leaves that cover intersections of the requested scope with the available leaf scopes.
    """
    leaves: list[Module] = []
    requested = set(scope.query)

    for leaf_module in leaf_modules:
        leaf_scope = set(leaf_module.scope.query)
        intersection = requested.intersection(leaf_scope)
        if not intersection:
            continue

        leaf = leaf_module.__class__(
            scope=Scope(sorted(intersection)), out_channels=leaf_module.out_shape.channels
        )
        leaf.maximum_likelihood_estimation(data)
        leaves.append(leaf)

    if not leaves:
        raise InvalidParameterError(
            f"No leaf module covers requested scope {tuple(scope.query)}. "
            f"Provide leaf_modules whose scopes cover all variables."
        )

    if len(scope.query) == 1:
        return leaves[0]
    if len(leaves) == 1:
        return Product(inputs=leaves[0])
    return Product(inputs=leaves)


def _learn_prometheus_single_channel(
    data: torch.Tensor,
    leaf_modules: list[LeafModule],
    scope: Scope,
    min_features_slice: int,
    min_instances_slice: int,
    n_clusters: int,
    clustering_method: Callable[..., torch.Tensor],
    affinity_method: Callable[[torch.Tensor], torch.Tensor],
    affinity_mode: str,
    similarity_kind: str,
    sampling_per_var: int | None,
    sampling_seed: int | None,
) -> Module:
    """Learn a single-channel Prometheus structure over a scope.

    Args:
        data: Full data tensor (batch, num_total_features).
        leaf_modules: Leaf module template(s) used for MLE initialization.
        scope: Scope for the current recursive call.
        min_features_slice: Minimum scope size before stopping recursion.
        min_instances_slice: Minimum number of samples before stopping recursion.
        n_clusters: Number of clusters for instance partitioning.
        clustering_method: Callable that returns cluster labels for scoped data.
        affinity_method: Callable that returns an affinity matrix for scoped data.
        affinity_mode: "full" or "sampled" affinity construction strategy.
        similarity_kind: "corr" or "rdc" when sampling edges; "custom" for full affinity only.
        sampling_per_var: Optional sampled neighbors per variable for sampled affinity.
        sampling_seed: Optional seed for deterministic neighbor sampling.

    Returns:
        A Module encoding the learned structure for this scope.
    """
    if len(scope.query) < min_features_slice or data.shape[0] < min_instances_slice:
        return _create_partitioned_mv_leaf(scope=scope, data=data, leaf_modules=leaf_modules)

    scoped_data = data[:, scope.query]
    labels = clustering_method(scoped_data, n_clusters=n_clusters)

    products: list[Module] = []
    product_priors: list[float] = []

    unique_clusters = torch.unique(labels)
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_data = data[cluster_mask, :]
        cluster_scoped = scoped_data[cluster_mask, :]

        if cluster_data.shape[0] < 2:
            continue

        if affinity_mode == "sampled":
            k = sampling_per_var
            if k is None:
                k = _default_samples_per_var(cluster_scoped.shape[1])
            if similarity_kind == "corr":
                edges = _sampled_edges_corr(cluster_scoped, k, sampling_seed)
            else:
                edges = _sampled_edges_rdc(cluster_scoped, k, sampling_seed)
            partitions = _mst_partitions_from_sampled_edges(cluster_scoped.shape[1], edges)
        else:
            affinity = affinity_method(cluster_scoped)
            partitions = _mst_partitions_from_affinity(affinity)
        if not partitions:
            continue

        # Memoize by scope to share subtrees across multiple partitions (DAG).
        memo: dict[tuple[int, ...], Module] = {}
        for partition in partitions:
            component_scopes = [Scope([scope.query[i] for i in component]) for component in partition]

            child_modules: list[Module] = []
            for child_scope in component_scopes:
                key = tuple(child_scope.query)
                if key not in memo:
                    memo[key] = _learn_prometheus_single_channel(
                        data=cluster_data,
                        leaf_modules=leaf_modules,
                        scope=child_scope,
                        min_features_slice=min_features_slice,
                        min_instances_slice=min_instances_slice,
                        n_clusters=n_clusters,
                        clustering_method=clustering_method,
                        affinity_method=affinity_method,
                        affinity_mode=affinity_mode,
                        similarity_kind=similarity_kind,
                        sampling_per_var=sampling_per_var,
                        sampling_seed=sampling_seed,
                    )
                child_modules.append(memo[key])

            product_inputs = _adapt_product_inputs(child_modules)
            products.append(Product(inputs=product_inputs))

        if partitions:
            cluster_prior = float(cluster_data.shape[0]) / float(data.shape[0])
            num_products_in_cluster = len(partitions)
            for _ in range(num_products_in_cluster):
                product_priors.append(cluster_prior / float(num_products_in_cluster))

    if not products:
        return _create_partitioned_mv_leaf(scope=scope, data=data, leaf_modules=leaf_modules)

    # Distribute mixture mass across product channels to match Sum's weight shape.
    per_channel_weights: list[float] = []
    for prior, product in zip(product_priors, products, strict=True):
        c = product.out_shape.channels
        per_channel_weights.extend([prior / float(c)] * c)

    total_weight = float(sum(per_channel_weights))
    if total_weight <= 0.0:
        return _create_partitioned_mv_leaf(scope=scope, data=data, leaf_modules=leaf_modules)
    per_channel_weights = [w / total_weight for w in per_channel_weights]

    # If only one product, keep model smaller (matches learn_spn behavior).
    if len(products) == 1:
        return products[0]

    return Sum(inputs=products, weights=per_channel_weights)


def learn_prometheus(
    data: torch.Tensor,
    leaf_modules: list[LeafModule] | LeafModule,
    out_channels: int = 1,
    min_features_slice: int = 2,
    min_instances_slice: int = 100,
    scope: Scope | None = None,
    n_clusters: int = 2,
    clustering_method: str | Callable[..., torch.Tensor] = "kmeans",
    similarity: str | Callable[[torch.Tensor], torch.Tensor] = "corr",
    affinity_mode: str = "full",
    sampling_per_var: int | None = None,
    sampling_seed: int | None = None,
    clustering_args: dict[str, Any] | None = None,
    similarity_args: dict[str, Any] | None = None,
) -> Module:
    """Learn an SPN structure using Prometheus.

    Args:
        data: 2D tensor (batch, num_total_features) containing training data.
        leaf_modules: Leaf module template(s) to use for fitting univariate/multivariate leaves.
        out_channels: Number of independent learned circuits (concatenated along channels).
        min_features_slice: Stop recursion when scope smaller than this.
        min_instances_slice: Stop recursion when fewer than this many samples at a node.
        scope: Scope to learn over. If None, inferred from leaf_modules.
        n_clusters: Number of instance clusters per recursion node (default 2).
        clustering_method: "kmeans" or a callable. Callable must accept `(scoped_data, n_clusters=...)`.
        similarity: "corr", "rdc", or a callable mapping `(scoped_cluster_data) -> affinity(d,d)`.
        affinity_mode: "full" (default) or "sampled" for the scalable approximation.
            Sampled mode supports only "corr" or "rdc" similarities.
        sampling_per_var: Number of sampled neighbors per variable when `affinity_mode="sampled"`.
            Defaults to `max(1, min(d-1, floor(log2(d))))` for each recursion scope.
        sampling_seed: Optional seed for deterministic neighbor sampling.
        clustering_args: Optional kwargs bound to clustering_method.
        similarity_args: Optional kwargs bound to similarity method.

    Returns:
        Learned SPN as a Module (may be a DAG with subtree sharing).

    Raises:
        InvalidTypeError: If arguments have invalid types.
        InvalidParameterError: If argument values are invalid or inconsistent.
        UnsupportedOperationError: If data contains NaNs (not supported for affinity computation).

    Pseudocode (Prometheus, from Jaini et al. 2018)::

        Prometheus(D, X):
            if |X| == 1: return univariate leaf fit on D
            cluster D into {D_i}; create sum node N
            for each D_i:
                build affinity matrix M_i and MST T_i
                while T_i has edges:
                    remove weakest edge from T_i
                    create product node P over connected components
                add all products from D_i as children of N
            for each product node P under N:
                for each scope S in P:
                    if sub-SPN for S not built:
                        build S via Prometheus(D restricted to S, S)
                    attach S as child of P
            return N
    """
    if scope is None:
        scope = _infer_scope_from_leaf_modules(leaf_modules)

    if isinstance(leaf_modules, LeafModule):
        leaf_modules = [leaf_modules]
    elif not isinstance(leaf_modules, list):
        raise InvalidTypeError(
            f"'leaf_modules' must be a LeafModule or list[LeafModule], got {type(leaf_modules)}."
        )

    if out_channels < 1:
        raise InvalidParameterError(f"'out_channels' must be >= 1, got {out_channels}.")
    if min_features_slice < 1:
        raise InvalidParameterError(f"'min_features_slice' must be >= 1, got {min_features_slice}.")
    if min_instances_slice < 1:
        raise InvalidParameterError(f"'min_instances_slice' must be >= 1, got {min_instances_slice}.")
    if n_clusters < 1:
        raise InvalidParameterError(f"'n_clusters' must be >= 1, got {n_clusters}.")
    if affinity_mode not in {"full", "sampled"}:
        raise InvalidParameterError(f"Unknown affinity_mode '{affinity_mode}'.")
    if sampling_per_var is not None and sampling_per_var < 1:
        raise InvalidParameterError(f"'sampling_per_var' must be >= 1, got {sampling_per_var}.")

    if isinstance(clustering_method, str):
        if clustering_method != "kmeans":
            raise InvalidParameterError(f"Unknown clustering_method '{clustering_method}'.")
        clustering_fn: Callable[..., torch.Tensor] = _cluster_by_kmeans
    else:
        clustering_fn = clustering_method

    if isinstance(similarity, str):
        if similarity == "corr":
            affinity_fn: Callable[[torch.Tensor], torch.Tensor] = _affinity_corr
            similarity_kind = "corr"
        elif similarity == "rdc":
            affinity_fn = _affinity_rdc
            similarity_kind = "rdc"
        else:
            raise InvalidParameterError(f"Unknown similarity '{similarity}'.")
    else:
        affinity_fn = similarity
        similarity_kind = "custom"

    if affinity_mode == "sampled" and similarity_kind == "custom":
        raise InvalidParameterError(
            "Sampled affinity only supports similarity='corr' or 'rdc' (custom callables are unsupported)."
        )

    if clustering_args is not None:
        clustering_fn = partial(clustering_fn, **clustering_args)
    if similarity_args is not None:
        affinity_fn = partial(affinity_fn, **similarity_args)

    sum_vectors: list[Module] = []
    for _ in range(out_channels):
        sum_vectors.append(
            _learn_prometheus_single_channel(
                data=data,
                leaf_modules=leaf_modules,
                scope=scope,
                min_features_slice=min_features_slice,
                min_instances_slice=min_instances_slice,
                n_clusters=n_clusters,
                clustering_method=clustering_fn,
                affinity_method=affinity_fn,
                affinity_mode=affinity_mode,
                similarity_kind=similarity_kind,
                sampling_per_var=sampling_per_var,
                sampling_seed=sampling_seed,
            )
        )

    if len(sum_vectors) == 1:
        return sum_vectors[0]
    return Cat(sum_vectors, dim=2)
