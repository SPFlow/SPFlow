import math

import torch

from spflow.measures.weight_of_evidence import (
    conditional_probability,
    weight_of_evidence,
    weight_of_evidence_leave_one_out,
)
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.categorical import Categorical
from spflow.modules.products.product import Product
from spflow.modules.sums.sum import Sum


def _det_cat(scope: int, value: int) -> Categorical:
    probs = torch.tensor([1.0, 0.0], dtype=torch.get_default_dtype())
    if value == 1:
        probs = torch.tensor([0.0, 1.0], dtype=torch.get_default_dtype())
    probs = probs.reshape(1, 1, 1, 2)
    return Categorical(scope=Scope([scope]), K=2, probs=probs)


def _joint_model_yx(weights: dict[tuple[int, int], float]) -> Sum:
    # Variables: y at index 0, x at index 1.
    comps = []
    w_list = []
    for (y, x), w in weights.items():
        comps.append(Product([_det_cat(0, y), _det_cat(1, x)]))
        w_list.append(w)
    return Sum(comps, weights=w_list)


def test_conditional_probability_matches_manual_ratio():
    # p(y=1,x=1)=0.45, p(y=0,x=1)=0.05 => p(y=1|x=1)=0.9
    model = _joint_model_yx({(1, 1): 0.45, (1, 0): 0.05, (0, 1): 0.05, (0, 0): 0.45})
    evidence = torch.tensor([[torch.nan, 1.0]], dtype=torch.get_default_dtype())
    p = conditional_probability(
        model, y_index=0, y_value=1, evidence=evidence, channel_agg="first", repetition_agg="first"
    )
    assert p.shape == (1,)
    assert abs(p.item() - 0.9) < 1e-6


def test_woe_matches_manual_log_odds_difference():
    model = _joint_model_yx({(1, 1): 0.45, (1, 0): 0.05, (0, 1): 0.05, (0, 0): 0.45})
    evidence_full = torch.tensor([[torch.nan, 1.0]], dtype=torch.get_default_dtype())
    evidence_reduced = torch.tensor([[torch.nan, torch.nan]], dtype=torch.get_default_dtype())

    n = 10_000
    k = 2
    w = weight_of_evidence(
        model,
        y_index=0,
        y_value=1,
        evidence_full=evidence_full,
        evidence_reduced=evidence_reduced,
        n=n,
        k=k,
        channel_agg="first",
        repetition_agg="first",
    )

    p_full = 0.9
    p_reduced = 0.5
    p_full_l = (p_full * n + 1) / (n + k)
    p_reduced_l = (p_reduced * n + 1) / (n + k)
    expected = math.log(p_full_l / (1 - p_full_l)) - math.log(p_reduced_l / (1 - p_reduced_l))

    assert w.shape == (1,)
    assert abs(w.item() - expected) < 1e-4

    w_rev = weight_of_evidence(
        model,
        y_index=0,
        y_value=1,
        evidence_full=evidence_reduced,
        evidence_reduced=evidence_full,
        n=n,
        k=k,
        channel_agg="first",
        repetition_agg="first",
    )
    assert abs((w + w_rev).item()) < 1e-6


def test_woe_leave_one_out_scores_only_observed_features():
    model = _joint_model_yx({(1, 1): 0.45, (1, 0): 0.05, (0, 1): 0.05, (0, 0): 0.45})
    x_instance = torch.tensor([[torch.nan, 1.0]], dtype=torch.get_default_dtype())
    we = weight_of_evidence_leave_one_out(
        model,
        y_index=0,
        y_value=1,
        x_instance=x_instance,
        n=10_000,
        k=2,
        channel_agg="first",
        repetition_agg="first",
    )
    assert we.shape == x_instance.shape
    assert torch.isnan(we[0, 0])
    assert torch.isfinite(we[0, 1])
