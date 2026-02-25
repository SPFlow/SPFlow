import pytest
import torch

from spflow.exceptions import InvalidParameterError, ShapeError
from spflow.zoo.hclt import learn_hclt_binary
from spflow.zoo.hclt.learn import _build_hclt_from_tree, _edges_to_adjacency, learn_hclt_categorical


def test_learn_hclt_binary_builds_and_scores() -> None:
    torch.manual_seed(0)
    data = torch.randint(0, 2, (32, 8), dtype=torch.float32)

    model = learn_hclt_binary(data, num_hidden_cats=3, num_trees=2, init="uniform")
    ll = model.log_likelihood(data)

    # HCLT learner returns a single-root density model with scalar log-likelihood per sample.
    assert tuple(ll.shape) == (32, 1, 1, 1)


def test_hclt_tree_helper_validation_errors() -> None:
    with pytest.raises(InvalidParameterError):
        _edges_to_adjacency(3, [(0, 3)])
    with pytest.raises(InvalidParameterError):
        _edges_to_adjacency(3, [(1, 1)])
    with pytest.raises(InvalidParameterError):
        _build_hclt_from_tree(
            num_features=2,
            edges=[(0, 1)],
            num_hidden_cats=0,
            emission_factory=lambda _: learn_hclt_binary(torch.randint(0, 2, (2, 2)), num_hidden_cats=1),
            init="uniform",
            device=None,
            dtype=None,
        )
    with pytest.raises(InvalidParameterError):
        _build_hclt_from_tree(
            num_features=3,
            edges=[(0, 1)],
            num_hidden_cats=2,
            emission_factory=lambda _: learn_hclt_binary(torch.randint(0, 2, (2, 2)), num_hidden_cats=1),
            init="uniform",
            device=None,
            dtype=None,
        )


@pytest.mark.parametrize("fn", [learn_hclt_binary, learn_hclt_categorical])
def test_hclt_input_validation_errors(fn) -> None:
    data = torch.randint(0, 2, (8, 4), dtype=torch.float32)
    with pytest.raises(ShapeError):
        fn(data.unsqueeze(0), num_hidden_cats=2)
    with pytest.raises(InvalidParameterError):
        # NaNs should be rejected early because MI estimation cannot recover from them.
        fn(data.masked_fill(torch.eye(8, 4, dtype=torch.bool), float("nan")), num_hidden_cats=2)
    with pytest.raises(InvalidParameterError):
        fn(data, num_hidden_cats=2, init="bad-init")
    with pytest.raises(InvalidParameterError):
        fn(data, num_hidden_cats=2, num_trees=0)


def test_learn_hclt_categorical_num_cats_inference_and_validation() -> None:
    data = torch.tensor([[0.0, 1.0], [2.0, 1.0], [1.0, 0.0]])
    model = learn_hclt_categorical(
        data,
        num_hidden_cats=2,
        num_trees=1,
        init="uniform",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert tuple(model.log_likelihood(data).shape) == (3, 1, 1, 1)

    with pytest.raises(InvalidParameterError):
        learn_hclt_categorical(torch.empty((0, 2), dtype=torch.float32), num_hidden_cats=2, num_cats=None)


def test_hclt_binary_and_categorical_device_dtype_branches() -> None:
    torch.manual_seed(1)
    xb = torch.randint(0, 2, (24, 6), dtype=torch.float32)
    xc = torch.randint(0, 4, (24, 6), dtype=torch.float32)
    device = torch.device("cpu")
    dtype = torch.float32

    mb = learn_hclt_binary(xb, num_hidden_cats=3, num_trees=2, init="uniform", device=device, dtype=dtype)
    mc = learn_hclt_categorical(
        xc, num_hidden_cats=3, num_cats=4, num_trees=2, init="uniform", device=device, dtype=dtype
    )

    assert tuple(mb.log_likelihood(xb).shape) == (24, 1, 1, 1)
    assert tuple(mc.log_likelihood(xc).shape) == (24, 1, 1, 1)


def test_learn_hclt_binary_single_tree_path() -> None:
    data = torch.randint(0, 2, (10, 4), dtype=torch.float32)
    model = learn_hclt_binary(data, num_hidden_cats=2, num_trees=1, init="uniform")
    assert tuple(model.log_likelihood(data).shape) == (10, 1, 1, 1)
