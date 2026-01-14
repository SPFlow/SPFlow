"""Hidden Chow-Liu Trees (HCLT) learner.

This builds a full probabilistic circuit whose structure is derived from a
Chow-Liu tree over the observed variables, and whose hidden states are modeled
via the channel dimension (H hidden categories per observed variable).
"""

from __future__ import annotations

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt.chow_liu import learn_chow_liu_trees_binary, learn_chow_liu_trees_categorical
from spflow.meta import Scope
from spflow.modules.leaves import Bernoulli, Categorical
from spflow.modules.products import ElementwiseProduct
from spflow.modules.sums import Sum
from spflow.modules.module import Module
from spflow.zoo.hclt.topk_mst import Edge


def _edges_to_adjacency(num_nodes: int, edges: list[Edge]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in edges:
        if a < 0 or b < 0 or a >= num_nodes or b >= num_nodes:
            raise InvalidParameterError("Edge indices out of range.")
        if a == b:
            raise InvalidParameterError("Self-edges are not allowed.")
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _build_hclt_from_tree(
    *,
    num_features: int,
    edges: list[Edge],
    num_hidden_cats: int,
    emission_factory,
    init: str,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Module:
    if num_hidden_cats < 1:
        raise InvalidParameterError("num_hidden_cats must be >= 1.")
    if len(edges) != num_features - 1:
        raise InvalidParameterError("Tree must have exactly num_features-1 edges.")

    adj = _edges_to_adjacency(num_features, edges)
    root = 0

    def uniform_sum_weights(in_ch: int, out_ch: int) -> Tensor:
        w = torch.full((1, in_ch, out_ch, 1), 1.0 / float(in_ch))
        if device is not None:
            w = w.to(device=device)
        if dtype is not None:
            w = w.to(dtype=dtype)
        return w

    def build_subtree(node: int, parent: int) -> Module:
        emission = emission_factory(node)

        child_msgs: list[Module] = []
        for ch in adj[node]:
            if ch == parent:
                continue
            child_sub = build_subtree(ch, node)
            weights = None if init != "uniform" else uniform_sum_weights(num_hidden_cats, num_hidden_cats)
            trans = Sum(inputs=child_sub) if weights is None else Sum(inputs=child_sub, weights=weights)
            child_msgs.append(trans)

        if not child_msgs:
            return emission
        return ElementwiseProduct(inputs=[emission, *child_msgs])

    subtree = build_subtree(root, -1)

    prior_weights = None if init != "uniform" else uniform_sum_weights(num_hidden_cats, 1)
    root_sum = Sum(inputs=subtree) if prior_weights is None else Sum(inputs=subtree, weights=prior_weights)
    return root_sum


def learn_hclt_binary(
    data: Tensor,
    *,
    num_hidden_cats: int,
    num_trees: int = 1,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    init: str = "uniform",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Module:
    """Learn an HCLT circuit from binary data.

    Args:
        data: (N, F) tensor with values in {0,1} (or bool). Must be complete (no NaNs).
        num_hidden_cats: Hidden categories per observed variable.
        num_trees: If >1, builds a mixture of HCLTs over the top-k Chow-Liu trees.
        dropout_prob: Edge dropout probability for top-k enumeration.
        weights: Optional per-sample weights.
        pseudocount: MI pseudocount (ChowLiuTrees.jl semantics).
        init: "uniform" or "random" (random uses module defaults).
        device/dtype: Optional placement overrides for created modules.
    """
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise InvalidParameterError("learn_hclt_binary requires complete data (no NaNs).")
    if init not in ("uniform", "random"):
        raise InvalidParameterError("init must be 'uniform' or 'random'.")
    if num_trees < 1:
        raise InvalidParameterError("num_trees must be >= 1.")

    num_features = int(data.shape[1])
    trees = learn_chow_liu_trees_binary(
        data,
        num_trees=num_trees,
        dropout_prob=dropout_prob,
        weights=weights,
        pseudocount=pseudocount,
    )

    def emission_factory(var: int) -> Module:
        leaf = Bernoulli(scope=Scope([var]), out_channels=num_hidden_cats)
        if device is not None:
            leaf = leaf.to(device=device)
        if dtype is not None:
            leaf = leaf.to(dtype=dtype)
        return leaf

    hclts = [
        _build_hclt_from_tree(
            num_features=num_features,
            edges=edges,
            num_hidden_cats=num_hidden_cats,
            emission_factory=emission_factory,
            init=init,
            device=device,
            dtype=dtype,
        )
        for edges in trees
    ]

    if len(hclts) == 1:
        return hclts[0]

    # Mixture over top-k HCLTs (learnable mixture weights).
    mix_weights = None
    if init == "uniform":
        mix_weights = torch.full((1, len(hclts), 1, 1), 1.0 / float(len(hclts)))
        if device is not None:
            mix_weights = mix_weights.to(device=device)
        if dtype is not None:
            mix_weights = mix_weights.to(dtype=dtype)
    return Sum(inputs=hclts) if mix_weights is None else Sum(inputs=hclts, weights=mix_weights)


def learn_hclt_categorical(
    data: Tensor,
    *,
    num_hidden_cats: int,
    num_cats: int | None = None,
    num_trees: int = 1,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    init: str = "uniform",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    chunk_size_pairs: int = 4096,
) -> Module:
    """Learn an HCLT circuit from categorical data.

    The structure is learned via a Chow-Liu tree on the observed variables, and
    emissions are `Categorical(X_i | Z_i)` with `num_hidden_cats` latent states.
    """
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if torch.isnan(data).any():
        raise InvalidParameterError("learn_hclt_categorical requires complete data (no NaNs).")
    if init not in ("uniform", "random"):
        raise InvalidParameterError("init must be 'uniform' or 'random'.")
    if num_trees < 1:
        raise InvalidParameterError("num_trees must be >= 1.")

    if num_cats is None:
        num_cats = int(data.max().item()) + 1 if data.numel() else 0
    if num_cats <= 0:
        raise InvalidParameterError("num_cats must be >= 1.")

    num_features = int(data.shape[1])
    trees = learn_chow_liu_trees_categorical(
        data,
        num_cats=num_cats,
        num_trees=num_trees,
        dropout_prob=dropout_prob,
        weights=weights,
        pseudocount=pseudocount,
        chunk_size_pairs=chunk_size_pairs,
    )

    def emission_factory(var: int) -> Module:
        leaf = Categorical(scope=Scope([var]), out_channels=num_hidden_cats, K=num_cats)
        if device is not None:
            leaf = leaf.to(device=device)
        if dtype is not None:
            leaf = leaf.to(dtype=dtype)
        return leaf

    hclts = [
        _build_hclt_from_tree(
            num_features=num_features,
            edges=edges,
            num_hidden_cats=num_hidden_cats,
            emission_factory=emission_factory,
            init=init,
            device=device,
            dtype=dtype,
        )
        for edges in trees
    ]

    if len(hclts) == 1:
        return hclts[0]

    mix_weights = None
    if init == "uniform":
        mix_weights = torch.full((1, len(hclts), 1, 1), 1.0 / float(len(hclts)))
        if device is not None:
            mix_weights = mix_weights.to(device=device)
        if dtype is not None:
            mix_weights = mix_weights.to(dtype=dtype)
    return Sum(inputs=hclts) if mix_weights is None else Sum(inputs=hclts, weights=mix_weights)
