"""Static guardrails for strict sampling-context contract."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_no_internal_init_default_sampling_context_calls() -> None:
    offenders: list[str] = []
    for path in list((REPO_ROOT / "spflow" / "modules").rglob("*.py")) + list(
        (REPO_ROOT / "spflow" / "zoo").rglob("*.py")
    ):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == "spflow/utils/sampling_context.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "init_default_sampling_context(" in text:
            offenders.append(rel)
    assert offenders == []


def test_weighted_sum_has_no_uniform_zero_row_fallback() -> None:
    text = _read("spflow/zoo/pic/weighted_sum.py")
    assert "torch.where(denom > 0, weights / denom" not in text


def test_split_by_index_has_no_repeat_truncate_heuristic() -> None:
    text = _read("spflow/modules/ops/split_by_index.py")
    assert "Truncate if we repeated too much" not in text
    assert "channel_index = channel_index[:, :input_features]" not in text


def test_conv_utils_expand_helper_removed() -> None:
    text = _read("spflow/modules/conv/utils.py")
    assert "def expand_sampling_context(" not in text


def test_sampling_p2_helpers_are_used_in_migrated_modules() -> None:
    expected_calls = {
        "spflow/modules/sums/sum.py": ["require_feature_width("],
        "spflow/modules/conv/sum_conv.py": ["require_feature_width("],
        "spflow/modules/conv/prod_conv.py": ["require_feature_width("],
        "spflow/modules/ops/cat.py": ["slice_feature_ranges(", "route_channel_offsets("],
        "spflow/modules/ops/split.py": ["require_feature_width("],
        "spflow/modules/ops/split_consecutive.py": ["repeat_split_feature_width("],
        "spflow/modules/ops/split_by_index.py": ["scatter_split_groups_to_input_width("],
        "spflow/modules/ops/split_interleaved.py": ["repeat_split_feature_width("],
        "spflow/modules/products/product.py": ["broadcast_feature_width("],
        "spflow/modules/products/base_product.py": ["require_feature_width("],
    }
    offenders: list[str] = []
    for rel, required_snippets in expected_calls.items():
        text = _read(rel)
        missing = [snippet for snippet in required_snippets if snippet not in text]
        if missing:
            offenders.append(f"{rel}: missing {missing}")
    assert offenders == []


def test_sampling_p2_bans_reintroduced_ad_hoc_feature_adaptation() -> None:
    banned_snippets_by_file = {
        "spflow/modules/ops/split_consecutive.py": ['"b f -> b (f s)"'],
        "spflow/modules/ops/split_by_index.py": ["new_zeros((data.shape[0], input_features))"],
        "spflow/modules/products/product.py": ['repeat(sampling_ctx.mask, "b 1 -> b f"'],
        "spflow/modules/ops/cat.py": ["global_channel_index = sampling_ctx.channel_index"],
    }
    offenders: list[str] = []
    for rel, banned_snippets in banned_snippets_by_file.items():
        text = _read(rel)
        for snippet in banned_snippets:
            if snippet in text:
                offenders.append(f"{rel}: contains banned snippet `{snippet}`")
                break
    assert offenders == []
