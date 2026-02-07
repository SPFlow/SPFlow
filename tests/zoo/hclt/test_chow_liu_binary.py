import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt.chow_liu import (
    learn_chow_liu_tree_binary,
    learn_chow_liu_tree_categorical,
    learn_chow_liu_trees_binary,
    learn_chow_liu_trees_categorical,
)
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


def test_learn_chow_liu_binary_validation_errors() -> None:
    x = torch.randint(0, 2, (8, 4))
    with pytest.raises(ShapeError):
        learn_chow_liu_trees_binary(x.unsqueeze(0))
    with pytest.raises(InvalidParameterError):
        learn_chow_liu_trees_binary(x, num_trees=0)
    with pytest.raises(InvalidParameterError):
        learn_chow_liu_trees_binary(x, pseudocount=-1.0)


def test_learn_chow_liu_categorical_validation_errors() -> None:
    x = torch.randint(0, 3, (8, 4))
    with pytest.raises(ShapeError):
        learn_chow_liu_trees_categorical(x.unsqueeze(0))
    with pytest.raises(InvalidParameterError):
        learn_chow_liu_trees_categorical(x, num_trees=0)
    with pytest.raises(InvalidParameterError):
        learn_chow_liu_trees_categorical(x, pseudocount=-1.0)


def test_learn_chow_liu_single_tree_wrappers() -> None:
    xb = torch.randint(0, 2, (24, 5), dtype=torch.long)
    tree_b = learn_chow_liu_tree_binary(xb)
    assert len(tree_b) == 4

    xc = torch.randint(0, 4, (24, 5), dtype=torch.long)
    tree_c = learn_chow_liu_tree_categorical(xc, num_cats=4)
    assert len(tree_c) == 4
