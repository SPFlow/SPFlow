import copy

import pytest

import torch

from spflow.dsl import term
from spflow.exceptions import ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.utils.compatibility import check_compatible_components


def test_compatibility_accepts_same_structure_different_parameters():
    a0 = Normal(scope=Scope([0]), out_channels=1, loc=torch.tensor([[[0.0]]]), scale=torch.tensor([[[1.0]]]))
    a1 = Normal(scope=Scope([1]), out_channels=1, loc=torch.tensor([[[1.0]]]), scale=torch.tensor([[[2.0]]]))
    b0 = Normal(scope=Scope([0]), out_channels=1, loc=torch.tensor([[[0.5]]]), scale=torch.tensor([[[1.5]]]))
    b1 = Normal(scope=Scope([1]), out_channels=1, loc=torch.tensor([[[1.2]]]), scale=torch.tensor([[[0.8]]]))

    m1 = Product([a0, a1])
    m2 = Product([b0, b1])
    check_compatible_components([m1, m2])


def test_compatibility_rejects_leaf_type_mismatch():
    a0 = Normal(scope=Scope([0]), out_channels=1)
    a1 = Normal(scope=Scope([1]), out_channels=1)
    m1 = Product([a0, a1])

    b0 = Bernoulli(scope=Scope([0]), out_channels=1)
    b1 = Normal(scope=Scope([1]), out_channels=1)
    m2 = Product([b0, b1])

    with pytest.raises(StructureError):
        check_compatible_components([m1, m2])


def test_compatibility_rejects_order_mismatch_in_cat_dim1():
    # Product([x0, x1]) uses Cat(dim=1) internally, so order matters.
    x0a = Normal(scope=Scope([0]), out_channels=1)
    x1a = Normal(scope=Scope([1]), out_channels=1)
    x0b = Normal(scope=Scope([0]), out_channels=1)
    x1b = Normal(scope=Scope([1]), out_channels=1)

    m1 = Product([x0a, x1a])
    m2 = Product([x1b, x0b])
    with pytest.raises(ShapeError):
        check_compatible_components([m1, m2])


def test_compatibility_rejects_cat_dim_mismatch():
    # Sum over two same-scope leaves uses Cat(dim=2); compare with a single-leaf Sum.
    a = Sum(inputs=[Normal(scope=0), Normal(scope=0)], out_channels=1, num_repetitions=1)
    b = Sum(inputs=Normal(scope=0), out_channels=1, num_repetitions=1)
    with pytest.raises(StructureError):
        check_compatible_components([a, b])


# Additional branch-focused compatibility tests
import torch

from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.ops.cat import Cat
from spflow.utils import compatibility as compatibility_mod
from spflow.utils.compatibility import CompatibilityIssue


def _normal(scope: int, channels: int = 1):
    return Normal(scope=Scope([scope]), out_channels=channels)


def test_compatibility_issue_str_and_short_circuit_return():
    issue = CompatibilityIssue(path="root", message="boom")
    assert str(issue) == "root: boom"

    # len < 2 should return early
    check_compatible_components([_normal(0)])


def test_compatibility_issue_str_and_short_list_return():
    issue = CompatibilityIssue(path="root", message="problem")
    assert str(issue) == "root: problem"
    check_compatible_components([Normal(scope=Scope([0]), out_channels=1)])


def test_check_pair_respects_visited_guard():
    a = _normal(0)
    b = _normal(0)
    visited = {(id(a), id(b))}

    compatibility_mod._check_pair(a, b, path="root", visited=visited)


def test_out_shape_cat_dim_cat_arity_and_k_mismatch_branches():
    with pytest.raises(ShapeError):
        compatibility_mod._check_pair(
            _normal(0, channels=1), _normal(0, channels=2), path="root", visited=set()
        )

    # Single-input Cat keeps same out_shape across dims; only dim differs.
    c1 = Cat([_normal(0)], dim=1)
    c2 = Cat([_normal(0)], dim=2)
    with pytest.raises(StructureError):
        compatibility_mod._check_pair(c1, c2, path="root", visited=set())

    # Different arity but same overall out_shape/channels.
    c3 = Cat([_normal(0, channels=1), _normal(0, channels=2)], dim=2)
    c4 = Cat([_normal(0, channels=3)], dim=2)
    with pytest.raises(StructureError):
        compatibility_mod._check_pair(c3, c4, path="root", visited=set())

    with pytest.raises(ShapeError):
        compatibility_mod._check_pair(
            Categorical(scope=Scope([0]), out_channels=1, K=2),
            Categorical(scope=Scope([0]), out_channels=1, K=3),
            path="root",
            visited=set(),
        )


def test_cltree_specific_branches_and_child_count_mismatch(monkeypatch):
    p = torch.tensor([-1, 0], dtype=torch.long)

    with pytest.raises(ShapeError):
        compatibility_mod._check_pair(
            CLTree(scope=Scope([0, 1]), out_channels=1, K=2, parents=p),
            CLTree(scope=Scope([0, 1]), out_channels=1, K=3, parents=p),
            path="root",
            visited=set(),
        )

    with pytest.raises(StructureError):
        compatibility_mod._check_pair(
            CLTree(scope=Scope([0, 1]), out_channels=1, K=2, parents=torch.tensor([-1, 0])),
            CLTree(scope=Scope([0, 1]), out_channels=1, K=2, parents=torch.tensor([-1, 1])),
            path="root",
            visited=set(),
        )

    a = _normal(0)
    b = _normal(0)

    def _fake_children(module):
        return [module] if module is a else []

    monkeypatch.setattr(compatibility_mod, "_children", _fake_children)
    with pytest.raises(StructureError):
        compatibility_mod._check_pair(a, b, path="root", visited=set())


def test_compatibility_rejects_out_shape_cat_and_leaf_parameter_mismatches():
    with pytest.raises(ShapeError):
        check_compatible_components(
            [
                Normal(scope=Scope([0]), out_channels=1, num_repetitions=1),
                Normal(scope=Scope([0]), out_channels=2, num_repetitions=1),
            ]
        )

    base = Sum(inputs=[Normal(scope=0), Normal(scope=0)], out_channels=1, num_repetitions=1)
    cat_dim2 = base.inputs
    cat_dim1 = copy.deepcopy(base.inputs)
    cat_dim1.dim = 1
    with pytest.raises(StructureError):
        check_compatible_components([cat_dim2, cat_dim1])

    c1 = Categorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=2,
        probs=torch.tensor([[[[0.4, 0.6]]]]),
    )
    c2 = Categorical(
        scope=Scope([0]),
        out_channels=1,
        num_repetitions=1,
        K=3,
        probs=torch.tensor([[[[0.2, 0.3, 0.5]]]]),
    )
    with pytest.raises(ShapeError):
        check_compatible_components([c1, c2])


def test_compatibility_rejects_cltree_mismatches():
    data = torch.randint(0, 2, (64, 2)).float()
    cl1 = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=2)
    cl2 = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=2)
    cl1.maximum_likelihood_estimation(data)
    cl2.maximum_likelihood_estimation(data)
    cl2.parents = torch.tensor([-1, -1], dtype=cl2.parents.dtype, device=cl2.parents.device)
    with pytest.raises(StructureError):
        check_compatible_components([cl1, cl2])

    cl3 = CLTree(scope=Scope([0, 1]), out_channels=1, num_repetitions=1, K=3)
    cl3.maximum_likelihood_estimation(data)
    with pytest.raises(ShapeError):
        check_compatible_components([cl1, cl3])


def test_compatibility_private_visited_and_child_count_branches(monkeypatch):
    a = Normal(scope=Scope([0]), out_channels=1)
    b = Normal(scope=Scope([0]), out_channels=1)
    visited = {(id(a), id(b))}
    compatibility_mod._check_pair(a, b, path="root", visited=visited)

    def _children(module):
        return [a] if module is a else []

    monkeypatch.setattr(compatibility_mod, "_children", _children)
    with pytest.raises(StructureError):
        compatibility_mod._check_pair(a, b, path="root", visited=set())


def test_compatibility_rejects_cat_arity_mismatch():
    c0 = Cat(inputs=[Normal(scope=0, out_channels=1), Normal(scope=0, out_channels=1)], dim=2)
    c1 = Cat(inputs=[Normal(scope=0, out_channels=2)], dim=2)
    with pytest.raises(StructureError):
        check_compatible_components([c0, c1])
