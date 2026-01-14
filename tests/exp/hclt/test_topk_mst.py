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
