from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from spflow.exceptions import InvalidParameterError
from spflow.measures._utils import infer_discrete_domains, reduce_log_likelihood
from spflow.meta.data.scope import Scope
from spflow.modules.module import Module


def conditional_probability(
    model: Module,
    *,
    y_index: int,
    y_value: int | float,
    evidence: Tensor,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Compute p(y=y_value | evidence) for a discrete target variable.

    This follows the legacy SPFlow definition:
        p(y|x) = p(x,y) / p(x)

    Args:
        model: SPFlow probabilistic circuit.
        y_index: Column index of the target variable Y in the data.
        y_value: Concrete value for Y.
        evidence: Evidence tensor of shape (batch, D) with NaNs for missing values.
        channel_agg: How to aggregate multiple channels ("logmeanexp", "logsumexp", "first").
        repetition_agg: How to aggregate multiple repetitions ("logmeanexp", "logsumexp", "first").

    Returns:
        Tensor of shape (batch,) with conditional probabilities in [0, 1].
    """
    if evidence.dim() != 2:
        raise InvalidParameterError(f"evidence must be 2D (batch, D), got shape {tuple(evidence.shape)}.")

    joint = evidence.clone()
    joint[:, y_index] = torch.as_tensor(y_value, dtype=joint.dtype, device=joint.device)

    denom = evidence.clone()
    denom[:, y_index] = torch.nan

    ll_joint = reduce_log_likelihood(
        model.log_likelihood(joint),
        channel_agg=channel_agg,
        repetition_agg=repetition_agg,
    )
    ll_denom = reduce_log_likelihood(
        model.log_likelihood(denom),
        channel_agg=channel_agg,
        repetition_agg=repetition_agg,
    )
    log_p = ll_joint - ll_denom
    return torch.exp(log_p)


def weight_of_evidence(
    model: Module,
    *,
    y_index: int,
    y_value: int | float,
    evidence_full: Tensor,
    evidence_reduced: Tensor,
    n: int,
    k: int | None = None,
    eps: float = 1e-6,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Compute the weight of evidence (WoE) between two evidence settings (in nats).

    This compares evidence_full against evidence_reduced using a log-odds difference:
        WoE = logit(L(p(y|e_full))) - logit(L(p(y|e_reduced)))

    where L(.) is a Laplace correction:
        L(p) = (p*n + 1) / (n + k)

    Args:
        model: SPFlow probabilistic circuit.
        y_index: Column index of Y.
        y_value: Concrete value for Y.
        evidence_full: Evidence tensor (batch, D).
        evidence_reduced: Evidence tensor (batch, D).
        n: Number of training instances used for Laplace correction.
        k: Cardinality of Y (if None, inferred for Bernoulli/Categorical).
        eps: Clamp used to keep probabilities away from 0/1 before logit.
        channel_agg: How to aggregate multiple channels ("logmeanexp", "logsumexp", "first").
        repetition_agg: How to aggregate multiple repetitions ("logmeanexp", "logsumexp", "first").

    Returns:
        Tensor of shape (batch,) with WoE values in nats.
    """
    if evidence_full.shape != evidence_reduced.shape:
        raise InvalidParameterError(
            f"evidence_full and evidence_reduced must have the same shape, got "
            f"{tuple(evidence_full.shape)} and {tuple(evidence_reduced.shape)}."
        )
    if n < 1:
        raise InvalidParameterError("n must be >= 1 for Laplace correction.")

    if k is None:
        domains = infer_discrete_domains(model, Scope([y_index]))
        k = int(domains[y_index].numel())

    p1 = conditional_probability(
        model,
        y_index=y_index,
        y_value=y_value,
        evidence=evidence_full,
        channel_agg=channel_agg,
        repetition_agg=repetition_agg,
    )
    p2 = conditional_probability(
        model,
        y_index=y_index,
        y_value=y_value,
        evidence=evidence_reduced,
        channel_agg=channel_agg,
        repetition_agg=repetition_agg,
    )

    n_t = torch.as_tensor(float(n), dtype=p1.dtype, device=p1.device)
    k_t = torch.as_tensor(float(k), dtype=p1.dtype, device=p1.device)
    p1_l = (p1 * n_t + 1.0) / (n_t + k_t)
    p2_l = (p2 * n_t + 1.0) / (n_t + k_t)

    p1_l = p1_l.clamp(min=eps, max=1.0 - eps)
    p2_l = p2_l.clamp(min=eps, max=1.0 - eps)
    return torch.logit(p1_l) - torch.logit(p2_l)


def weight_of_evidence_leave_one_out(
    model: Module,
    *,
    y_index: int,
    y_value: int | float,
    x_instance: Tensor,
    n: int,
    k: int | None = None,
    eps: float = 1e-6,
    channel_agg: str = "logmeanexp",
    repetition_agg: str = "logmeanexp",
) -> Tensor:
    """Compute per-feature leave-one-out WoE attributions (legacy-style, in nats).

    For each non-NaN entry X_i in ``x_instance`` (excluding ``y_index``), this computes:
        WoE_i = logit(L(p(y|x))) - logit(L(p(y|x\\i)))

    Args:
        model: SPFlow probabilistic circuit.
        y_index: Column index of Y.
        y_value: Concrete value for Y.
        x_instance: Evidence tensor of shape (batch, D). NaNs indicate missing values.
        n: Number of training instances used for Laplace correction.
        k: Cardinality of Y (if None, inferred for Bernoulli/Categorical).
        eps: Clamp used to keep probabilities away from 0/1 before logit.
        channel_agg: How to aggregate multiple channels ("logmeanexp", "logsumexp", "first").
        repetition_agg: How to aggregate multiple repetitions ("logmeanexp", "logsumexp", "first").

    Returns:
        Tensor of shape (batch, D) with WoE scores per feature and NaNs elsewhere.
    """
    if x_instance.dim() != 2:
        raise InvalidParameterError(f"x_instance must be 2D (batch, D), got shape {tuple(x_instance.shape)}.")

    base = x_instance.clone()
    base[:, y_index] = torch.as_tensor(y_value, dtype=base.dtype, device=base.device)

    out = x_instance.clone()
    out[:, y_index] = torch.nan

    # mask of features to score (non-NaN and not y_index)
    score_mask = ~torch.isnan(out)
    score_mask[:, y_index] = False

    if score_mask.sum() == 0:
        return out

    for j in range(out.shape[1]):
        if j == y_index:
            continue
        if not bool(score_mask[:, j].any()):
            continue

        reduced = base.clone()
        reduced[:, j] = torch.nan

        w = weight_of_evidence(
            model,
            y_index=y_index,
            y_value=y_value,
            evidence_full=base,
            evidence_reduced=reduced,
            n=n,
            k=k,
            eps=eps,
            channel_agg=channel_agg,
            repetition_agg=repetition_agg,
        )

        out[:, j] = w

    return out


__all__ = [
    "conditional_probability",
    "weight_of_evidence",
    "weight_of_evidence_leave_one_out",
]
