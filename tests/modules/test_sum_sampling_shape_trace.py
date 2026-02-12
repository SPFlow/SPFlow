import torch

from spflow.modules.sums import Sum
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext
from tests.utils.leaves import make_normal_leaf


def _make_sum(in_channels: int, out_channels: int, out_features: int, num_repetitions: int = 1) -> Sum:
    inputs = make_normal_leaf(
        out_features=out_features,
        out_channels=in_channels,
        num_repetitions=num_repetitions,
    )
    return Sum(inputs=inputs, out_channels=out_channels, num_repetitions=num_repetitions)


def _trace_sum_sample_shapes(module: Sum, sampling_ctx: SamplingContext, cache: Cache | None, is_mpe: bool) -> dict[str, tuple[int, ...]]:
    """Trace and print intermediate tensor shapes from Sum.sample logic."""
    shapes: dict[str, tuple[int, ...]] = {}

    print(f"TRACE sampling_ctx.channel_index: {tuple(sampling_ctx.channel_index.shape)}")
    print(f"TRACE sampling_ctx.mask:          {tuple(sampling_ctx.mask.shape)}")

    if sampling_ctx.repetition_idx is not None:
        logits = module.logits.unsqueeze(0).expand(
            sampling_ctx.channel_index.shape[0], -1, -1, -1, -1
        )
        shapes["logits_expand_repetition"] = tuple(logits.shape)
        print(f"TRACE logits_expand_repetition:  {shapes['logits_expand_repetition']}")

        in_channels_total = logits.shape[2]
        indices = sampling_ctx.repetition_idx.view(-1, 1, 1, 1, 1).expand(
            -1, logits.shape[1], in_channels_total, logits.shape[3], -1
        )
        shapes["repetition_indices"] = tuple(indices.shape)
        print(f"TRACE repetition_indices:        {shapes['repetition_indices']}")

        logits = torch.gather(logits, dim=-1, index=indices).squeeze(-1)
    else:
        logits = module.logits[..., 0].unsqueeze(0)
        shapes["logits_base"] = tuple(logits.shape)
        print(f"TRACE logits_base:               {shapes['logits_base']}")

        logits = logits.expand(sampling_ctx.channel_index.shape[0], -1, -1, -1)
        shapes["logits_expand_batch"] = tuple(logits.shape)
        print(f"TRACE logits_expand_batch:       {shapes['logits_expand_batch']}")

    idxs = sampling_ctx.channel_index[..., None, None]
    shapes["idxs_initial"] = tuple(idxs.shape)
    print(f"TRACE idxs_initial:              {shapes['idxs_initial']}")

    in_channels_total = logits.shape[2]
    idxs = idxs.expand(-1, -1, in_channels_total, -1)
    shapes["idxs_expand"] = tuple(idxs.shape)
    print(f"TRACE idxs_expand:               {shapes['idxs_expand']}")

    logits = logits.gather(dim=3, index=idxs).squeeze(3)
    shapes["logits_after_parent_index"] = tuple(logits.shape)
    print(f"TRACE logits_after_parent_index: {shapes['logits_after_parent_index']}")

    if cache is not None and "log_likelihood" in cache and cache["log_likelihood"].get(module.inputs) is not None:
        input_lls = cache["log_likelihood"][module.inputs]
        shapes["input_lls_raw"] = tuple(input_lls.shape)
        print(f"TRACE input_lls_raw:             {shapes['input_lls_raw']}")

        if input_lls.dim() == 4 and input_lls.shape[-1] == 1:
            input_lls = input_lls.squeeze(-1)
            shapes["input_lls_squeezed"] = tuple(input_lls.shape)
            print(f"TRACE input_lls_squeezed:        {shapes['input_lls_squeezed']}")

        logits = (logits + input_lls).log_softmax(dim=2)
        shapes["logits_after_conditioning"] = tuple(logits.shape)
        print(f"TRACE logits_after_conditioning: {shapes['logits_after_conditioning']}")

    if is_mpe:
        new_channel_index = torch.argmax(logits, dim=-1)
    else:
        new_channel_index = torch.distributions.Categorical(logits=logits).sample()

    shapes["new_channel_index"] = tuple(new_channel_index.shape)
    print(f"TRACE new_channel_index:         {shapes['new_channel_index']}")

    if new_channel_index.shape != sampling_ctx.mask.shape:
        new_mask = sampling_ctx.mask.expand_as(new_channel_index).contiguous()
        shapes["new_mask"] = tuple(new_mask.shape)
        print(f"TRACE new_mask_expanded:         {shapes['new_mask']}")
    else:
        print("TRACE new_mask_expanded:         <no expansion>")

    return shapes


def test_sum_sample_shape_trace_unconditional():
    """Print and validate shapes for unconditional Sum sampling (no cache)."""
    torch.manual_seed(7)
    batch_size = 3
    out_features = 4
    in_channels = 2
    out_channels = 3

    module = _make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=1,
    )
    data = torch.full((batch_size, out_features), torch.nan)
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((batch_size, out_features), dtype=torch.long),
        mask=torch.ones((batch_size, out_features), dtype=torch.bool),
    )

    shapes = _trace_sum_sample_shapes(module, sampling_ctx=sampling_ctx, cache=None, is_mpe=False)
    assert shapes["new_channel_index"] == (batch_size, out_features)

    out = module.sample(data=data, sampling_ctx=sampling_ctx)
    assert out.shape == (batch_size, out_features)
    assert sampling_ctx.channel_index.shape == (batch_size, out_features)
    assert sampling_ctx.mask.shape == (batch_size, out_features)


def test_sum_sample_shape_trace_conditional_expands_feature_axis():
    """Print and validate shapes for conditional Sum sampling under strict feature-width context."""
    torch.manual_seed(11)
    batch_size = 3
    out_features = 4
    in_channels = 2
    out_channels = 3

    module = _make_sum(
        in_channels=in_channels,
        out_channels=out_channels,
        out_features=out_features,
        num_repetitions=1,
    )

    evidence = torch.randn(batch_size, out_features)
    cache = Cache()
    _ = module.log_likelihood(evidence, cache=cache)

    data = torch.full((batch_size, out_features), torch.nan)
    sampling_ctx = SamplingContext(
        channel_index=torch.zeros((batch_size, out_features), dtype=torch.long),
        mask=torch.ones((batch_size, out_features), dtype=torch.bool),
    )

    shapes = _trace_sum_sample_shapes(module, sampling_ctx=sampling_ctx, cache=cache, is_mpe=False)
    assert shapes["input_lls_squeezed"] == (batch_size, out_features, in_channels)
    assert shapes["new_channel_index"] == (batch_size, out_features)

    out = module.sample(data=data, cache=cache, sampling_ctx=sampling_ctx)
    assert out.shape == (batch_size, out_features)
    assert sampling_ctx.channel_index.shape == (batch_size, out_features)
    assert sampling_ctx.mask.shape == (batch_size, out_features)
