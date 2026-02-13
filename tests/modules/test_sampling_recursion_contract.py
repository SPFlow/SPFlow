"""Static recursion-contract audits for differentiable sampling paths."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TARGETED_MODULE_PATHS = (
    "spflow/modules/sums/sum.py",
    "spflow/modules/sums/elementwise_sum.py",
    "spflow/modules/sums/signed_sum.py",
    "spflow/modules/sums/repetition_mixing_layer.py",
    "spflow/modules/einsum/linsum_layer.py",
    "spflow/modules/einsum/einsum_layer.py",
    "spflow/modules/conv/prod_conv.py",
    "spflow/modules/conv/sum_conv.py",
    "spflow/modules/products/product.py",
    "spflow/modules/products/base_product.py",
    "spflow/modules/rat/factorize.py",
    "spflow/modules/ops/split.py",
    "spflow/modules/ops/split_consecutive.py",
    "spflow/modules/ops/split_interleaved.py",
    "spflow/modules/ops/split_by_index.py",
    "spflow/modules/ops/cat.py",
    "spflow/modules/wrapper/base.py",
    "spflow/modules/wrapper/image_wrapper.py",
    "spflow/modules/leaves/leaf.py",
    "spflow/zoo/apc/encoders/convpc_joint_encoder.py",
    "spflow/zoo/cms/joint.py",
    "spflow/zoo/conv/conv_pc.py",
    "spflow/zoo/pic/weighted_sum.py",
    "spflow/zoo/sos/models.py",
)


def _collect_forbidden_calls(
    module_path: Path,
    *,
    fn_name: str,
    forbidden_attr: str,
) -> list[int]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    bad_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            for nested in ast.walk(node):
                if (
                    isinstance(nested, ast.Call)
                    and isinstance(nested.func, ast.Attribute)
                    and nested.func.attr == forbidden_attr
                ):
                    bad_lines.append(nested.lineno)
    return bad_lines


@pytest.mark.parametrize("module_relpath", _TARGETED_MODULE_PATHS)
def test_targeted_rsample_never_calls_sample(module_relpath: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / module_relpath

    bad_lines = _collect_forbidden_calls(
        module_path,
        fn_name="_rsample",
        forbidden_attr="_sample",
    )

    assert not bad_lines, (
        f"{module_relpath} violates recursion contract: `_rsample` must not call `_sample` "
        f"(lines: {bad_lines})."
    )


@pytest.mark.parametrize("module_relpath", _TARGETED_MODULE_PATHS)
def test_targeted_sample_never_calls_rsample(module_relpath: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / module_relpath

    bad_lines = _collect_forbidden_calls(
        module_path,
        fn_name="_sample",
        forbidden_attr="_rsample",
    )

    assert not bad_lines, (
        f"{module_relpath} violates recursion contract: `_sample` must not call `_rsample` "
        f"(lines: {bad_lines})."
    )
