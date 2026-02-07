import copy

import pytest

import torch

from spflow.dsl import term
from spflow.exceptions import ShapeError, StructureError
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.bernoulli import Bernoulli
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.leaves.cltree import CLTree
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.ops.cat import Cat
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum
from spflow.zoo.sos.compatibility import CompatibilityIssue
from spflow.zoo.sos import compatibility as compatibility_mod
from spflow.zoo.sos import check_compatible_components


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


def test_compatibility_issue_str_and_short_list_return():
    issue = CompatibilityIssue(path="root", message="problem")
    assert str(issue) == "root: problem"
    check_compatible_components([Normal(scope=Scope([0]), out_channels=1)])


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

    class Dummy(Module):
        @property
        def feature_to_scope(self):
            return a.feature_to_scope

        def log_likelihood(self, data, cache=None):
            raise NotImplementedError

        def sample(self, num_samples=None, data=None, is_mpe=False, cache=None, sampling_ctx=None):
            raise NotImplementedError

        def expectation_maximization(self, data, bias_correction=True, cache=None):
            raise NotImplementedError

        def maximum_likelihood_estimation(self, data, weights=None, cache=None):
            raise NotImplementedError

        def marginalize(self, marg_rvs, prune=True, cache=None):
            raise NotImplementedError

    d1 = Dummy()
    d2 = Dummy()
    d1.scope = Scope([0])
    d2.scope = Scope([0])
    d1.out_shape = a.out_shape
    d2.out_shape = a.out_shape

    monkeypatch.setattr(
        compatibility_mod,
        "_children",
        lambda module: [a] if module is d1 else [],
    )
    with pytest.raises(StructureError):
        compatibility_mod._check_pair(d1, d2, path="root", visited=set())


def test_compatibility_rejects_cat_arity_mismatch():
    c0 = Cat(inputs=[Normal(scope=0, out_channels=1), Normal(scope=0, out_channels=1)], dim=2)
    c1 = Cat(inputs=[Normal(scope=0, out_channels=2)], dim=2)
    with pytest.raises(StructureError):
        check_compatible_components([c0, c1])
