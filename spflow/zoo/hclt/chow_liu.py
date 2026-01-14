"""Chow-Liu tree learning utilities.

These helpers provide ChowLiuTrees.jl-style structure learning (including top-k)
for binary data. The output is an edge list with 0-based variable indices.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt.mi import pairwise_mi_binary, pairwise_mi_categorical
from spflow.zoo.hclt.topk_mst import Edge, topk_mst


def learn_chow_liu_trees_binary(
    data: Tensor,
    *,
    num_trees: int = 1,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> list[list[Edge]]:
    """Learn top-k Chow-Liu trees from binary data (0/1 or bool)."""
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if num_trees < 1:
        raise InvalidParameterError("num_trees must be >= 1.")
    if pseudocount < 0:
        raise InvalidParameterError("pseudocount must be >= 0.")

    mi = pairwise_mi_binary(data, weights=weights, pseudocount=pseudocount, dtype=dtype)
    # Maximum spanning tree on MI == minimum spanning tree on -MI.
    dist = (-mi).detach()
    if dist.is_cuda:
        dist = dist.cpu()
    dist_np = dist.numpy().astype(np.float64, copy=False)

    return topk_mst(dist_np, num_trees=num_trees, dropout_prob=dropout_prob, minimize=True)


def learn_chow_liu_tree_binary(
    data: Tensor,
    *,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> list[Edge]:
    """Learn a single Chow-Liu tree (best MST) from binary data."""
    trees = learn_chow_liu_trees_binary(
        data,
        num_trees=1,
        dropout_prob=dropout_prob,
        weights=weights,
        pseudocount=pseudocount,
        dtype=dtype,
    )
    return trees[0]


def learn_chow_liu_trees_categorical(
    data: Tensor,
    *,
    num_cats: int | None = None,
    num_trees: int = 1,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    dtype: torch.dtype = torch.float64,
    chunk_size_pairs: int = 4096,
) -> list[list[Edge]]:
    """Learn top-k Chow-Liu trees from categorical data (values 0..K-1)."""
    if data.dim() != 2:
        raise ShapeError(f"data must be 2D (N,F), got shape {tuple(data.shape)}.")
    if num_trees < 1:
        raise InvalidParameterError("num_trees must be >= 1.")
    if pseudocount < 0:
        raise InvalidParameterError("pseudocount must be >= 0.")

    mi = pairwise_mi_categorical(
        data,
        num_cats=num_cats,
        weights=weights,
        pseudocount=pseudocount,
        dtype=dtype,
        chunk_size_pairs=chunk_size_pairs,
    )
    dist = (-mi).detach()
    if dist.is_cuda:
        dist = dist.cpu()
    dist_np = dist.numpy().astype(np.float64, copy=False)
    return topk_mst(dist_np, num_trees=num_trees, dropout_prob=dropout_prob, minimize=True)


def learn_chow_liu_tree_categorical(
    data: Tensor,
    *,
    num_cats: int | None = None,
    dropout_prob: float = 0.0,
    weights: Tensor | None = None,
    pseudocount: float = 1.0,
    dtype: torch.dtype = torch.float64,
    chunk_size_pairs: int = 4096,
) -> list[Edge]:
    trees = learn_chow_liu_trees_categorical(
        data,
        num_cats=num_cats,
        num_trees=1,
        dropout_prob=dropout_prob,
        weights=weights,
        pseudocount=pseudocount,
        dtype=dtype,
        chunk_size_pairs=chunk_size_pairs,
    )
    return trees[0]
