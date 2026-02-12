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
