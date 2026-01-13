import math

import torch

from spflow.measures.information_theory import entropy, mutual_information
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum


def test_entropy_matches_known_distribution():
    probs = torch.tensor([0.25, 0.75], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    model = Categorical(scope=Scope([0]), K=2, probs=probs)

    h = entropy(model, Scope([0]), method="exact", channel_agg="first", repetition_agg="first")
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert torch.isfinite(h)
    assert h.ndim == 0
    assert abs(h.item() - expected) < 1e-6


def test_mutual_information_zero_for_independent_model():
    probs_x = torch.tensor([0.1, 0.9], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    probs_y = torch.tensor([0.4, 0.6], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)

    # Duplicate parameters across two channels to exercise aggregation code paths.
    probs_x = probs_x.expand(1, 2, 1, 2).contiguous()
    probs_y = probs_y.expand(1, 2, 1, 2).contiguous()

    x = Categorical(scope=Scope([0]), K=2, probs=probs_x)
    y = Categorical(scope=Scope([1]), K=2, probs=probs_y)
    model = Product([x, y])

    mi = mutual_information(model, Scope([0]), Scope([1]), method="exact")
    assert torch.isfinite(mi)
    assert abs(mi.item()) < 1e-6


def test_mutual_information_positive_for_dependent_model():
    # Perfect correlation via a 50/50 mixture:
    # p(x=0,y=0)=0.5, p(x=1,y=1)=0.5.
    p0 = torch.tensor([1.0, 0.0], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    p1 = torch.tensor([0.0, 1.0], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    x0 = Categorical(scope=Scope([0]), K=2, probs=p0)
    y0 = Categorical(scope=Scope([1]), K=2, probs=p0)
    x1 = Categorical(scope=Scope([0]), K=2, probs=p1)
    y1 = Categorical(scope=Scope([1]), K=2, probs=p1)

    comp0 = Product([x0, y0])
    comp1 = Product([x1, y1])
    model = Sum([comp0, comp1], weights=[0.5, 0.5])

    mi = mutual_information(model, Scope([0]), Scope([1]), method="exact")
    assert torch.isfinite(mi)
    assert mi.item() > 0.5
    assert abs(mi.item() - math.log(2.0)) < 1e-6
