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


def test_check_pair_respects_visited_guard():
    a = _normal(0)
    b = _normal(0)
    visited = {(id(a), id(b))}

    compatibility_mod._check_pair(a, b, path="root", visited=visited)


def test_out_shape_cat_dim_cat_arity_and_k_mismatch_branches():
    with pytest.raises(ShapeError, match="out_shape mismatch"):
        compatibility_mod._check_pair(_normal(0, channels=1), _normal(0, channels=2), path="root", visited=set())

    # Single-input Cat keeps same out_shape across dims; only dim differs.
    c1 = Cat([_normal(0)], dim=1)
    c2 = Cat([_normal(0)], dim=2)
    with pytest.raises(StructureError, match="Cat dim mismatch"):
        compatibility_mod._check_pair(c1, c2, path="root", visited=set())

    # Different arity but same overall out_shape/channels.
    c3 = Cat([_normal(0, channels=1), _normal(0, channels=2)], dim=2)
    c4 = Cat([_normal(0, channels=3)], dim=2)
    with pytest.raises(StructureError, match="Cat arity mismatch"):
        compatibility_mod._check_pair(c3, c4, path="root", visited=set())

    with pytest.raises(ShapeError, match="Categorical K mismatch"):
        compatibility_mod._check_pair(
            Categorical(scope=Scope([0]), out_channels=1, K=2),
            Categorical(scope=Scope([0]), out_channels=1, K=3),
            path="root",
            visited=set(),
        )


def test_cltree_specific_branches_and_child_count_mismatch(monkeypatch):
    p = torch.tensor([-1, 0], dtype=torch.long)

    with pytest.raises(ShapeError, match="CLTree K mismatch"):
        compatibility_mod._check_pair(
            CLTree(scope=Scope([0, 1]), out_channels=1, K=2, parents=p),
            CLTree(scope=Scope([0, 1]), out_channels=1, K=3, parents=p),
            path="root",
            visited=set(),
        )

    with pytest.raises(StructureError, match="CLTree parents mismatch"):
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
    with pytest.raises(StructureError, match="child count mismatch"):
        compatibility_mod._check_pair(a, b, path="root", visited=set())
