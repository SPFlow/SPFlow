import pytest

import torch

from spflow.dsl import term
from spflow.exceptions import ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.exp.sos import check_compatible_components


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
