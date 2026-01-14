import pytest
import torch

from spflow.zoo.hclt.mi import pairwise_mi_binary, pairwise_marginal_binary


def test_pairwise_mi_binary_matches_chowliutrees_jl_values() -> None:
    # Ported from ChowLiuTrees.jl/test/information_tests.jl
    x = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    mi = pairwise_mi_binary(x, pseudocount=0.0)

    assert mi[0, 0].item() == pytest.approx(0.6931471805599453)
    assert mi[0, 1].item() == pytest.approx(0.08630462173553435)
    assert mi[0, 2].item() == pytest.approx(0.0)
    assert mi[1, 1].item() == pytest.approx(0.6730116670092565)
    assert mi[1, 2].item() == pytest.approx(0.013844293808390695)
    assert mi[2, 2].item() == pytest.approx(0.6730116670092565)


def test_pairwise_marginal_binary_shapes() -> None:
    x = torch.randint(0, 2, (17, 5))
    pxy = pairwise_marginal_binary(x, pseudocount=1.0)
    assert tuple(pxy.shape) == (5, 5, 4)
