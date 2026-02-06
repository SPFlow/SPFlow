import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt.mi import (
    _as_float_tensor,
    pairwise_marginal_binary,
    pairwise_mi_binary,
    pairwise_mi_categorical,
)


def test_pairwise_marginal_binary_validation_errors() -> None:
    x = torch.randint(0, 2, (6, 3), dtype=torch.float32)

    with pytest.raises(ShapeError, match="must be 2D"):
        pairwise_marginal_binary(x.unsqueeze(0))
    with pytest.raises(InvalidParameterError, match="pseudocount must be >= 0"):
        pairwise_marginal_binary(x, pseudocount=-1.0)
    with pytest.raises(ShapeError, match="weights must have shape"):
        pairwise_marginal_binary(x, weights=torch.ones(6, 1))
    with pytest.raises(InvalidParameterError, match="weights must be finite"):
        pairwise_marginal_binary(x, weights=torch.tensor([1.0, 1.0, float("inf"), 1.0, 1.0, 1.0]))
    with pytest.raises(InvalidParameterError, match="Total mass"):
        pairwise_marginal_binary(x, weights=-torch.ones(6), pseudocount=0.0)


def test_pairwise_marginal_binary_weighted_path() -> None:
    x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float32)
    w = torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=torch.float32)
    pxy = pairwise_marginal_binary(x, weights=w, pseudocount=0.5, dtype=torch.float32)
    assert tuple(pxy.shape) == (2, 2, 4)
    assert torch.isfinite(pxy).all()


def test_as_float_tensor_device_conversion_branch() -> None:
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = _as_float_tensor(x, dtype=torch.float32, device=torch.device("meta"))
    assert y.device.type == "meta"


def test_pairwise_mi_binary_runs_with_weights() -> None:
    x = torch.randint(0, 2, (12, 4), dtype=torch.float32)
    w = torch.linspace(0.5, 1.5, 12, dtype=torch.float32)
    mi = pairwise_mi_binary(x, weights=w, pseudocount=0.25, dtype=torch.float32)
    assert tuple(mi.shape) == (4, 4)
    assert torch.isfinite(mi).all()


def test_pairwise_mi_categorical_validation_errors() -> None:
    x = torch.randint(0, 3, (7, 4), dtype=torch.float32)

    with pytest.raises(ShapeError, match="must be 2D"):
        pairwise_mi_categorical(x.unsqueeze(0), num_cats=3)
    with pytest.raises(InvalidParameterError, match="pseudocount must be >= 0"):
        pairwise_mi_categorical(x, num_cats=3, pseudocount=-1.0)
    with pytest.raises(InvalidParameterError, match="chunk_size_pairs must be >= 1"):
        pairwise_mi_categorical(x, num_cats=3, chunk_size_pairs=0)
    with pytest.raises(InvalidParameterError, match="no NaNs"):
        pairwise_mi_categorical(x.masked_fill(torch.eye(7, 4, dtype=torch.bool), float("nan")), num_cats=3)
    with pytest.raises(InvalidParameterError, match="num_cats must be >= 1"):
        pairwise_mi_categorical(torch.empty((0, 2), dtype=torch.float32), num_cats=None)
    with pytest.raises(InvalidParameterError, match="Categorical data must be in"):
        pairwise_mi_categorical(torch.tensor([[0, 1], [2, 3]], dtype=torch.float32), num_cats=3)
    with pytest.raises(ShapeError, match="weights must have shape"):
        pairwise_mi_categorical(x, num_cats=3, weights=torch.ones(7, 1))
    with pytest.raises(InvalidParameterError, match="weights must be finite"):
        pairwise_mi_categorical(
            x, num_cats=3, weights=torch.tensor([1.0, 1.0, 1.0, float("inf"), 1.0, 1.0, 1.0])
        )
    with pytest.raises(InvalidParameterError, match="Total mass"):
        pairwise_mi_categorical(x, num_cats=3, weights=-torch.ones(7), pseudocount=0.0)


def test_pairwise_mi_categorical_weighted_and_single_feature_paths() -> None:
    x = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 2, 0], [1, 1, 2]], dtype=torch.float32)
    w = torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=torch.float32)

    mi = pairwise_mi_categorical(x, num_cats=3, weights=w, pseudocount=0.5, dtype=torch.float32)
    assert tuple(mi.shape) == (3, 3)
    assert torch.isfinite(mi).all()

    mi_single = pairwise_mi_categorical(torch.tensor([[0], [1], [2]], dtype=torch.float32), num_cats=3)
    assert tuple(mi_single.shape) == (1, 1)
