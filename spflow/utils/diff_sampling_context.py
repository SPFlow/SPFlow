from __future__ import annotations

import torch
from torch import Tensor

from spflow.utils.diff_sampling import DiffSampleMethod
from spflow.utils.sampling_context import SamplingContext


class DifferentiableSamplingContext(SamplingContext):
    """Sampling context carrying extra tensors for differentiable routing."""

    def __init__(
        self,
        num_samples: int | None = None,
        device: torch.device | None = None,
        channel_index: Tensor | None = None,
        mask: Tensor | None = None,
        repetition_index: Tensor | None = None,
        *,
        is_differentiable: bool = True,
        method: DiffSampleMethod = DiffSampleMethod.SIMPLE,
        tau: float = 1.0,
        hard: bool = True,
        channel_select: Tensor | None = None,
        repetition_select: Tensor | None = None,
        branch_weight: Tensor | None = None,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            device=device,
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_index,
        )
        self.is_differentiable = is_differentiable
        self.method = method
        self.tau = tau
        self.hard = hard
        self.channel_select = channel_select
        self.repetition_select = repetition_select
        self.branch_weight = branch_weight

    def copy(self) -> "DifferentiableSamplingContext":
        return DifferentiableSamplingContext(
            channel_index=self.channel_index.clone(),
            mask=self.mask.clone(),
            repetition_index=self.repetition_idx.clone() if self.repetition_idx is not None else None,
            is_differentiable=self.is_differentiable,
            method=self.method,
            tau=self.tau,
            hard=self.hard,
            channel_select=self.channel_select.clone() if self.channel_select is not None else None,
            repetition_select=self.repetition_select.clone() if self.repetition_select is not None else None,
            branch_weight=self.branch_weight.clone() if self.branch_weight is not None else None,
        )
