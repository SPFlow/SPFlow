"""Top-k spanning trees for complete graphs.

This implements the dense, exact algorithm used in ChowLiuTrees.jl:
- Kruskal MST on a complete graph given a full weight matrix.
- Yamada-style enumeration of the top-k MSTs via constraint branching.

This is intentionally CPU-focused and expects dense NxN weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

from spflow.exceptions import InvalidParameterError, ShapeError

Edge = tuple[int, int]


def _canon_edge(e: Edge) -> Edge:
    a, b = e
    return (a, b) if a <= b else (b, a)


@dataclass
class _UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, n: int) -> "_UnionFind":
        return cls(parent=list(range(n)), rank=[0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def kruskal_mst_complete(weights: np.ndarray, *, minimize: bool = True) -> list[Edge]:
    """Compute an MST on a complete graph given dense weights."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ShapeError(f"weights must be square (N,N), got {weights.shape}.")
    n = int(weights.shape[0])
    if n <= 0:
        raise InvalidParameterError("weights must be non-empty.")
    if not np.isfinite(weights).all():
        # We allow +/-inf for constraints and dropout, but not NaN.
        if np.isnan(weights).any():
            raise InvalidParameterError("weights must not contain NaNs.")

    ii, jj = np.triu_indices(n, k=1)
    w = weights[ii, jj]
    # Match Julia `sortperm` tie-breaking for equal weights: stable ordering by
    # the original edge list order (i ascending, then j ascending).
    if minimize:
        order = np.lexsort((jj, ii, w))
    else:
        order = np.lexsort((jj, ii, -w))

    uf = _UnionFind.create(n)
    mst: list[Edge] = []
    for idx in order:
        a = int(ii[idx])
        b = int(jj[idx])
        if uf.union(a, b):
            mst.append((a, b))
            if len(mst) == n - 1:
                break
    return [_canon_edge(e) for e in mst]


def mst_with_constraints(
    weights: np.ndarray,
    included_edges: Iterable[Edge],
    excluded_edges: Iterable[Edge],
    *,
    reuse: np.ndarray,
    dropout_prob: float = 0.0,
    rng: np.random.Generator | None = None,
    minimize: bool = True,
) -> tuple[list[Edge] | None, float | None]:
    """Compute MST under include/exclude constraints (dense, complete graph)."""
    if dropout_prob < 0.0 or dropout_prob >= 1.0:
        raise InvalidParameterError("dropout_prob must be in [0,1).")
    if reuse.shape != weights.shape:
        raise ShapeError("reuse must have the same shape as weights.")

    reuse[:, :] = weights

    n = int(weights.shape[0])
    if dropout_prob > 0.0:
        rng = rng or np.random.default_rng()
        mask = rng.random((n, n)) < dropout_prob
        mask = np.triu(mask, k=1)
        reuse[mask] = np.inf
        reuse[(mask.T)] = np.inf

    included = {_canon_edge(e) for e in included_edges}
    excluded = {_canon_edge(e) for e in excluded_edges}

    for a, b in included:
        reuse[a, b] = -np.inf
        reuse[b, a] = -np.inf
    for a, b in excluded:
        reuse[a, b] = np.inf
        reuse[b, a] = np.inf

    mst = kruskal_mst_complete(reuse, minimize=minimize)
    mst_set = set(mst)

    if not included.issubset(mst_set):
        return None, None
    if mst_set.intersection(excluded):
        return None, None

    total = 0.0
    for a, b in mst:
        total += float(weights[a, b])
    return mst, total


def topk_mst(
    weights: np.ndarray,
    *,
    num_trees: int = 1,
    dropout_prob: float = 0.0,
    rng: np.random.Generator | None = None,
    minimize: bool = True,
) -> list[list[Edge]]:
    """Enumerate the top-k MSTs for a dense complete graph.

    This follows ChowLiuTrees.jl `topk_MST` (Yamada enumeration).
    """
    if num_trees < 1:
        raise InvalidParameterError("num_trees must be >= 1.")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ShapeError(f"weights must be square (N,N), got {weights.shape}.")

    reuse = np.empty_like(weights)

    mst0, total0 = mst_with_constraints(
        weights,
        included_edges=[],
        excluded_edges=[],
        reuse=reuse,
        dropout_prob=0.0,
        rng=rng,
        minimize=minimize,
    )
    if mst0 is None or total0 is None:
        return []

    # Tie-breaking for equal totals: lexicographic on (mst_edges, included, excluded)
    # for deterministic output.
    def key_for(mst_edges: list[Edge], included_edges: list[Edge], excluded_edges: list[Edge]) -> tuple:
        return (tuple(mst_edges), tuple(included_edges), tuple(excluded_edges))

    heap: list[tuple[float, tuple, list[Edge], list[Edge], list[Edge]]] = []
    heappush(heap, (float(total0), key_for(mst0, [], []), mst0, [], []))

    out: list[list[Edge]] = []
    while heap and len(out) < num_trees:
        total, _, mst_edges, included_edges, excluded_edges = heappop(heap)
        out.append(mst_edges)

        if len(out) == num_trees:
            break

        # Match ChowLiuTrees.jl branching exactly (mutating included/excluded lists
        # as we walk MST edges).
        included_edges = list(map(_canon_edge, included_edges))
        excluded_edges = list(map(_canon_edge, excluded_edges))
        included_set = set(included_edges)

        edge_added = False
        for edge in map(_canon_edge, mst_edges):
            if edge in included_set:
                continue

            if edge_added:
                # Move the previously excluded MST edge into included.
                included_edges.append(excluded_edges.pop())
                included_set.add(included_edges[-1])

            excluded_edges.append(edge)
            edge_added = True

            cand_mst, cand_total = mst_with_constraints(
                weights,
                included_edges,
                excluded_edges,
                reuse=reuse,
                dropout_prob=dropout_prob,
                rng=rng,
                minimize=minimize,
            )
            if cand_mst is None or cand_total is None:
                continue
            inc = list(included_edges)
            exc = list(excluded_edges)
            heappush(heap, (float(cand_total), key_for(cand_mst, inc, exc), cand_mst, inc, exc))

    return out
