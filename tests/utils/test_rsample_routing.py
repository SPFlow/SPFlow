import torch

from spflow.utils.diff_sampling_context import DifferentiableSamplingContext
from spflow.utils.rsample_routing import (
    condition_logits_with_evidence,
    ensure_diff_ctx,
    merge_disjoint_child_outputs,
    sample_selector_and_index,
    select_parent_axis,
    update_ctx_channel_routing,
)
from spflow.utils.sampling_context import SamplingContext


def test_select_parent_axis_soft_matches_hard_one_hot():
    tensor = torch.randn(3, 4, 5, 6)
    channel_index = torch.randint(0, 6, (3, 4))
    mask = torch.ones((3, 4), dtype=torch.bool)

    hard_ctx = SamplingContext(channel_index=channel_index.clone(), mask=mask.clone())
    hard = select_parent_axis(tensor, sampling_ctx=hard_ctx, dim=3)

    soft_ctx = SamplingContext(channel_index=channel_index.clone(), mask=mask.clone())
    soft_ctx.channel_select = torch.nn.functional.one_hot(channel_index, num_classes=6).to(tensor.dtype)
    soft = select_parent_axis(tensor, sampling_ctx=soft_ctx, dim=3)

    torch.testing.assert_close(hard, soft, rtol=0.0, atol=0.0)


def test_condition_logits_with_evidence_matches_manual_posterior():
    logits = torch.randn(2, 3, 4)
    evidence = torch.randn(2, 3, 4)

    conditioned = condition_logits_with_evidence(logits, evidence, dim=2)
    expected = (logits + evidence).log_softmax(dim=2)

    torch.testing.assert_close(conditioned, expected)
    probs = conditioned.exp().sum(dim=2)
    torch.testing.assert_close(probs, torch.ones_like(probs), rtol=1e-5, atol=1e-6)


def test_update_ctx_channel_routing_expands_mask_and_sets_metadata():
    ctx = DifferentiableSamplingContext(
        channel_index=torch.zeros((5, 1), dtype=torch.long),
        mask=torch.ones((5, 1), dtype=torch.bool),
    )

    new_index = torch.zeros((5, 3), dtype=torch.long)
    selector = torch.nn.functional.one_hot(new_index, num_classes=2).to(torch.get_default_dtype())

    update_ctx_channel_routing(
        ctx,
        channel_index=new_index,
        channel_select=selector,
        method="simple",
        tau=0.7,
        hard=True,
    )

    assert ctx.channel_index.shape == (5, 3)
    assert ctx.mask.shape == (5, 3)
    assert ctx.channel_select.shape == (5, 3, 2)
    assert ctx.tau == 0.7


def test_sample_selector_and_index_shapes_and_consistency():
    logits = torch.randn(4, 6, 5)
    selector, index = sample_selector_and_index(
        logits=logits,
        dim=-1,
        is_mpe=False,
        method="simple",
        tau=1.0,
        hard=True,
    )
    assert selector.shape == logits.shape
    assert index.shape == logits.shape[:-1]
    torch.testing.assert_close(index, selector.argmax(dim=-1))


def test_merge_disjoint_child_outputs_prefers_existing_then_fills_missing():
    base = torch.tensor([[float("nan"), 3.0, float("nan")]])
    left = torch.tensor([[1.0, 3.0, float("nan")]])
    right = torch.tensor([[float("nan"), 3.0, 9.0]])

    merged = merge_disjoint_child_outputs(base, [left, right])
    torch.testing.assert_close(merged, torch.tensor([[1.0, 3.0, 9.0]]))


def test_ensure_diff_ctx_promotes_sampling_context_and_copies_repetition():
    base_ctx = SamplingContext(
        channel_index=torch.zeros((2, 1), dtype=torch.long),
        mask=torch.ones((2, 1), dtype=torch.bool),
        repetition_index=torch.ones((2,), dtype=torch.long),
    )
    ctx = ensure_diff_ctx(
        base_ctx,
        batch_size=2,
        features=4,
        device=torch.device("cpu"),
        method="simple",
        tau=1.0,
        hard=True,
    )
    assert isinstance(ctx, DifferentiableSamplingContext)
    assert ctx.repetition_idx is not None
