import math

import pytest
import torch

from spflow.exceptions import InvalidParameterError
from spflow.measures.information_theory import (
    conditional_mutual_information,
    entropy,
    mutual_information,
)
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum


def test_entropy_matches_known_distribution():
    # Closed-form Bernoulli entropy provides a precise reference value.
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


def test_entropy_errors_and_mc_path():
    # Deterministic leaf guarantees zero entropy and zero MC variance.
    probs = torch.tensor([1.0, 0.0], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    model = Categorical(scope=Scope([0]), K=2, probs=probs)

    with pytest.raises(InvalidParameterError):
        entropy(model, Scope([]), method="exact")

    with pytest.raises(InvalidParameterError):
        entropy(model, Scope([0]), method="nope")

    with pytest.raises(InvalidParameterError):
        entropy(model, Scope([0]), method="mc", num_samples=0)

    # Same seed should yield identical scalar outputs through the forked RNG context.
    h_exact = entropy(model, Scope([0]), method="exact")
    h_mc_1 = entropy(model, Scope([0]), method="mc", num_samples=32, seed=7)
    h_mc_2 = entropy(model, Scope([0]), method="mc", num_samples=32, seed=7)

    assert h_exact.item() == 0.0
    assert h_mc_1.item() == 0.0
    assert h_mc_2.item() == 0.0
    assert torch.equal(h_mc_1, h_mc_2)


def test_mutual_information_errors_and_mc_path():
    # Independent deterministic variables must have zero MI for both estimators.
    p = torch.tensor([1.0, 0.0], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    x = Categorical(scope=Scope([0]), K=2, probs=p)
    y = Categorical(scope=Scope([1]), K=2, probs=p)
    model = Product([x, y])

    with pytest.raises(InvalidParameterError):
        mutual_information(model, Scope([0]), Scope([0]), method="exact")

    with pytest.raises(InvalidParameterError):
        mutual_information(model, Scope([0]), Scope([1]), method="unknown")

    with pytest.raises(InvalidParameterError):
        mutual_information(model, Scope([0]), Scope([1]), method="mc", num_samples=0)

    mi_mc = mutual_information(model, Scope([0]), Scope([1]), method="mc", num_samples=64, seed=4)
    assert torch.isfinite(mi_mc)
    assert mi_mc.item() == 0.0


def test_conditional_mutual_information_exact_and_mc_for_independent_model():
    # Product factorization enforces X ⟂ Y | Z (and in fact X ⟂ Y,Z), so CMI is zero.
    px = torch.tensor([0.3, 0.7], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    py = torch.tensor([0.6, 0.4], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    pz = torch.tensor([0.2, 0.8], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    x = Categorical(scope=Scope([0]), K=2, probs=px)
    y = Categorical(scope=Scope([1]), K=2, probs=py)
    z = Categorical(scope=Scope([2]), K=2, probs=pz)
    model = Product([x, y, z])

    cmi_exact = conditional_mutual_information(model, Scope([0]), Scope([1]), Scope([2]), method="exact")
    cmi_mc = conditional_mutual_information(
        model,
        Scope([0]),
        Scope([1]),
        Scope([2]),
        method="mc",
        num_samples=1024,
        seed=3,
    )
    assert torch.isfinite(cmi_exact)
    assert torch.isfinite(cmi_mc)
    assert abs(cmi_exact.item()) < 1e-6
    # MC estimator has sampling noise; keep a loose but meaningful tolerance.
    assert abs(cmi_mc.item()) < 5e-2


def test_conditional_mutual_information_errors():
    p = torch.tensor([0.5, 0.5], dtype=torch.get_default_dtype()).reshape(1, 1, 1, 2)
    x = Categorical(scope=Scope([0]), K=2, probs=p)
    y = Categorical(scope=Scope([1]), K=2, probs=p)
    z = Categorical(scope=Scope([2]), K=2, probs=p)
    model = Product([x, y, z])

    with pytest.raises(InvalidParameterError):
        conditional_mutual_information(model, Scope([0]), Scope([1]), Scope([0]), method="exact")

    with pytest.raises(InvalidParameterError):
        conditional_mutual_information(model, Scope([0]), Scope([1]), Scope([2]), method="other")

    with pytest.raises(InvalidParameterError):
        conditional_mutual_information(model, Scope([0]), Scope([1]), Scope([2]), method="mc", num_samples=0)
