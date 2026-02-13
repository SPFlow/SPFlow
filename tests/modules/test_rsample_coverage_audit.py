"""Coverage audit for class-level differentiable sampling hooks."""

from __future__ import annotations

import importlib
import pkgutil

import spflow
from spflow.modules.module import Module

_ALLOWED_DEFAULT_RSAMPLE: set[str] = set()


def _iter_all_subclasses(base: type) -> set[type]:
    out: set[type] = set()
    queue = [base]
    while queue:
        current = queue.pop()
        for child in current.__subclasses__():
            if child not in out:
                out.add(child)
                queue.append(child)
    return out


def test_module_subclasses_do_not_fall_back_to_default_rsample() -> None:
    for _, modname, _ in pkgutil.walk_packages(spflow.__path__, prefix="spflow."):
        try:
            importlib.import_module(modname)
        except Exception:  # pragma: no cover - guardrail only
            continue

    fallback_classes = sorted(
        f"{cls.__module__}.{cls.__name__}"
        for cls in _iter_all_subclasses(Module)
        if cls.__module__.startswith("spflow.")
        if getattr(cls, "_rsample", None) is Module._rsample
    )
    assert fallback_classes == sorted(_ALLOWED_DEFAULT_RSAMPLE), (
        "Unexpected Module subclasses falling back to Module._rsample: "
        f"{fallback_classes}. Allowed fallback list: {sorted(_ALLOWED_DEFAULT_RSAMPLE)}."
    )
