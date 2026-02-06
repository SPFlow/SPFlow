import numpy as np

from spflow.zoo.hclt.topk_mst import topk_mst


def test_topk_mst_matches_chowliutrees_jl_fixture() -> None:
    # Ported from ChowLiuTrees.jl/test/chow_liu_trees_tests.jl (0-based indices).
    mi = np.array(
        [
            [0, 1, 3, 4],
            [1, 0, 2, 5],
            [3, 2, 0, 6],
            [4, 5, 6, 0],
        ],
        dtype=np.float64,
    )

    msts1 = topk_mst(mi, num_trees=10, minimize=True)
    # ChowLiuTrees.jl's first 8 trees are unambiguous for this fixture.
    expected_prefix = [
        [(0, 1), (1, 2), (0, 3)],
        [(0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 2), (1, 3)],
        [(0, 1), (0, 2), (1, 3)],
        [(0, 1), (1, 2), (2, 3)],
        [(1, 2), (0, 2), (0, 3)],
        [(0, 1), (0, 2), (2, 3)],
        [(0, 2), (1, 2), (1, 3)],
    ]

    assert len(msts1) == 10
    assert msts1[:8] == expected_prefix

    # The last two slots have a three-way tie at total weight 11 for this graph.
    tie11 = {
        tuple([(0, 1), (0, 3), (2, 3)]),
        tuple([(0, 2), (1, 2), (2, 3)]),
        tuple([(0, 3), (1, 2), (1, 3)]),
    }
    last_two = {tuple(msts1[8]), tuple(msts1[9])}
    assert last_two.issubset(tie11)
    assert len(last_two) == 2


# Additional branch-focused topk_mst tests
import pytest

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt.topk_mst import kruskal_mst_complete, mst_with_constraints


def test_kruskal_validates_shape_and_nan():
    with pytest.raises(ShapeError):
        kruskal_mst_complete(np.ones((2, 3)))

    with pytest.raises(InvalidParameterError, match="non-empty"):
        kruskal_mst_complete(np.zeros((0, 0)))

    x = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(InvalidParameterError, match="must not contain NaNs"):
        kruskal_mst_complete(x)


def test_kruskal_maximize_branch():
    w = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )
    mst = kruskal_mst_complete(w, minimize=False)
    assert set(mst) == {(0, 2), (1, 2)}


def test_mst_with_constraints_validation_and_dropout():
    w = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    reuse = np.empty_like(w)

    with pytest.raises(InvalidParameterError, match=r"\[0,1\)"):
        mst_with_constraints(w, [], [], reuse=reuse, dropout_prob=1.0)

    with pytest.raises(ShapeError):
        mst_with_constraints(w, [], [], reuse=np.empty((2, 2)))

    rng = np.random.default_rng(0)
    mst, total = mst_with_constraints(w, [], [], reuse=reuse, dropout_prob=0.5, rng=rng)
    assert (mst is None and total is None) or (isinstance(mst, list) and isinstance(total, float))


def test_mst_constraints_return_none_for_infeasible_included_or_excluded():
    w = np.array(
        [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 1.0],
            [10.0, 1.0, 0.0],
        ]
    )
    reuse = np.empty_like(w)

    mst, total = mst_with_constraints(w, included_edges=[(0, 2)], excluded_edges=[], reuse=reuse)
    assert mst is not None and total is not None

    mst_bad, total_bad = mst_with_constraints(w, included_edges=[(0, 2)], excluded_edges=[(0, 2)], reuse=reuse)
    assert mst_bad is None and total_bad is None


def test_topk_validation_and_empty_seed_case(monkeypatch):
    with pytest.raises(InvalidParameterError):
        topk_mst(np.ones((2, 2)), num_trees=0)

    with pytest.raises(ShapeError):
        topk_mst(np.ones((2, 3)), num_trees=1)

    def _always_none(*args, **kwargs):
        return None, None

    monkeypatch.setattr("spflow.zoo.hclt.topk_mst.mst_with_constraints", _always_none)
    assert topk_mst(np.ones((2, 2)), num_trees=2) == []
