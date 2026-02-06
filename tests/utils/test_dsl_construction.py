"""Unit tests for the example-oriented DSL construction helpers."""

from dataclasses import dataclass

import pytest
import torch

from spflow.dsl import (
    SumExpr,
    WeightedExpr,
    _make_sum_weights,
    _validate_product_modules,
    _validate_sum_modules,
    as_expr,
    build,
    dsl,
    term,
    w,
)
from spflow.exceptions import (
    InvalidParameterCombinationError,
    InvalidParameterError,
    InvalidWeightsError,
    ScopeError,
    ShapeError,
)
from spflow.meta.data.scope import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.module_shape import ModuleShape
from spflow.modules.sums.sum import Sum


@dataclass
class _StubModule:
    scope: Scope
    out_shape: ModuleShape
    device: torch.device


class TestDslConstruction:
    """Tests for building modules via `spflow.dsl` expressions."""

    def test_weighted_sum_builds_valid_module(self):
        """Builds a simple weighted sum and runs log_likelihood."""
        a = term(Normal(scope=0))
        b = term(Normal(scope=0))

        expr = 0.4 * a + 0.6 * b
        model = expr.build()

        assert isinstance(model, Module)
        assert isinstance(model, Sum)
        assert len(model.scope.query) == 1
        assert model.out_shape.features == 1
        assert model.out_shape.channels == 1

        data = torch.randn(5, 1)
        ll = model.log_likelihood(data)
        assert ll.shape == (5, 1, 1, 1)

    def test_product_requires_disjoint_scopes(self):
        """Rejects products whose factors overlap in scope."""
        a = term(Normal(scope=0))
        b = term(Normal(scope=0))

        expr = a * b
        with pytest.raises(ScopeError):
            _ = expr.build()

    def test_nested_expression_builds_and_runs_log_likelihood(self):
        """Builds a small nested mixture/product circuit and runs inference."""
        x0 = term(Normal(scope=0))
        x1a = term(Normal(scope=1))
        x2a = term(Normal(scope=2))
        x1b = term(Normal(scope=1))
        x2b = term(Normal(scope=2))

        inner = 0.3 * (x1a * x2a) + 0.7 * (x1b * x2b)
        expr = 0.4 * (x0 * inner) + 0.6 * (x0 * x1a * x2a)
        model = expr.build()

        data = torch.randn(7, 3)
        ll = model.log_likelihood(data)
        assert ll.shape == (7, 1, 1, 1)

    def test_mpe_and_sample_work_on_dsl_built_model(self):
        """Sampling and MPE run end-to-end on a DSL-built model."""
        x0 = term(Normal(scope=0))
        x1 = term(Normal(scope=1))
        x2 = term(Normal(scope=2))

        expr = 0.4 * (x0 * x1 * x2) + 0.6 * (x0 * x1 * x2)
        model = expr.build()

        samples = model.sample(num_samples=5)
        assert samples.shape == (5, 3)

        mpe = model.mpe(num_samples=4)
        assert mpe.shape == (4, 3)

    def test_weights_are_normalized(self):
        """Weights are normalized on build (ergonomic behavior)."""
        a = term(Normal(scope=0))
        b = term(Normal(scope=0))

        model = (2.0 * a + 3.0 * b).build()
        assert isinstance(model, Sum)

        weights = model.weights
        assert weights.shape[0] == 1  # features
        assert weights.shape[1] == 2  # in_channels (one per term)
        assert weights.shape[2] == 1  # out_channels
        assert weights.shape[3] == 1  # repetitions

        torch.testing.assert_close(
            weights[0, :, 0, 0],
            torch.tensor([2.0 / 5.0, 3.0 / 5.0], dtype=weights.dtype, device=weights.device),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_unweighted_add_is_rejected(self):
        """Rejects `A + B` to force explicit weights in examples."""
        a = term(Normal(scope=0))
        b = term(Normal(scope=0))
        with pytest.raises(InvalidParameterError):
            _ = a + b  # type: ignore[operator]

    def test_sum_rejects_multi_channel_terms(self):
        """Rejects weighted sums when any term has out_channels > 1."""
        a = term(Normal(scope=0, out_channels=2))
        b = term(Normal(scope=0, out_channels=2))

        with pytest.raises(ShapeError):
            _ = (0.4 * a + 0.6 * b).build()

    def test_context_manager_allows_unwrapped_leaves_and_lmul(self):
        """`dsl()` enables `float * Module`, `Module * float`, and `Module * Module`."""
        with dsl():
            expr = 0.4 * Normal(0) * Normal(1) + (Normal(0) * Normal(1)) * 0.6
        model = expr.build()
        assert isinstance(model, Module)
        ll = model.log_likelihood(torch.randn(3, 2))
        assert ll.shape == (3, 1, 1, 1)


class TestDslHelpersAndErrors:
    def test_as_expr_rejects_invalid_type(self):
        with pytest.raises(InvalidParameterError):
            _ = as_expr("not-an-expr")  # type: ignore[arg-type]

    def test_w_and_build_helpers_cover_module_and_expr_paths(self):
        module = Normal(scope=0)
        expr = w(2.0, module) + w(3.0, term(Normal(scope=0)))

        passthrough = build(module)
        built = build(expr)

        assert passthrough is module
        assert isinstance(built, Sum)

    def test_term_and_product_rmul_reject_non_numeric_weights(self):
        assert isinstance(term(Normal(scope=0)) * 2.0, WeightedExpr)
        with pytest.raises(InvalidParameterError):
            _ = term(Normal(scope=0)).__rmul__("bad")  # type: ignore[arg-type]
        with pytest.raises(InvalidParameterError):
            _ = (term(Normal(scope=0)) * term(Normal(scope=1))).__rmul__("bad")  # type: ignore[arg-type]

    def test_weighted_expr_validation_and_build_errors(self):
        with pytest.raises(InvalidParameterError):
            _ = WeightedExpr(weight="bad", expr=term(Normal(scope=0)))  # type: ignore[arg-type]
        with pytest.raises(InvalidWeightsError):
            _ = WeightedExpr(weight=float("nan"), expr=term(Normal(scope=0)))
        with pytest.raises(InvalidWeightsError):
            _ = WeightedExpr(weight=0.0, expr=term(Normal(scope=0)))
        with pytest.raises(InvalidParameterError):
            _ = WeightedExpr(weight=1.0, expr=term(Normal(scope=0))).build()

    def test_weighted_expr_add_radd_and_invalid_add(self):
        a = w(1.0, term(Normal(scope=0)))
        b = w(2.0, term(Normal(scope=0)))
        c = w(3.0, term(Normal(scope=0)))

        left = a + b
        merged = c + left
        appended = left + c
        radded = c.__radd__(left)

        assert isinstance(merged, SumExpr)
        assert isinstance(appended, SumExpr)
        assert isinstance(radded, SumExpr)
        assert len(merged.terms) == 3
        assert len(appended.terms) == 3
        assert len(radded.terms) == 3
        assert c.__radd__(123) is NotImplemented  # type: ignore[comparison-overlap]

        with pytest.raises(InvalidParameterError):
            _ = a + 1  # type: ignore[operator]

    def test_sum_expr_validation_and_operator_errors(self):
        leaf = term(Normal(scope=0))
        with pytest.raises(InvalidParameterError):
            _ = SumExpr([(1.0, leaf)])
        with pytest.raises(InvalidWeightsError):
            _ = SumExpr([(1.0, leaf), (0.0, leaf)])

        valid = SumExpr([(1.0, leaf), (2.0, leaf)])
        other = SumExpr([(1.0, leaf), (3.0, leaf)])
        merged = valid + other
        weighted = valid * 2.0
        rmul_weighted = valid.__rmul__(2.0)
        product = valid * term(Normal(scope=1))
        assert isinstance(merged, SumExpr)
        assert isinstance(weighted, WeightedExpr)
        assert isinstance(rmul_weighted, WeightedExpr)
        assert isinstance(product.build(), Module)

        with pytest.raises(InvalidParameterError):
            _ = valid + 1  # type: ignore[operator]
        with pytest.raises(InvalidParameterError):
            _ = valid.__rmul__("bad")  # type: ignore[arg-type]

    def test_validate_product_modules_error_paths(self):
        base = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("cpu"))
        with pytest.raises(InvalidParameterError):
            _validate_product_modules([base])  # type: ignore[arg-type]

        a = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("cpu"))
        b_scope = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("cpu"))
        b_ch = _StubModule(Scope([1]), ModuleShape(1, 2, 1), torch.device("cpu"))
        b_rep = _StubModule(Scope([1]), ModuleShape(1, 1, 2), torch.device("cpu"))
        b_dev = _StubModule(Scope([1]), ModuleShape(1, 1, 1), torch.device("meta"))

        with pytest.raises(ScopeError):
            _validate_product_modules([a, b_scope])  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            _validate_product_modules([a, b_ch])  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            _validate_product_modules([a, b_rep])  # type: ignore[arg-type]
        with pytest.raises(InvalidParameterCombinationError):
            _validate_product_modules([a, b_dev])  # type: ignore[arg-type]

    def test_validate_sum_modules_error_paths(self):
        base = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("cpu"))
        with pytest.raises(InvalidParameterError):
            _validate_sum_modules([base])  # type: ignore[arg-type]

        a = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("cpu"))
        b_scope = _StubModule(Scope([1]), ModuleShape(1, 1, 1), torch.device("cpu"))
        b_feat = _StubModule(Scope([0]), ModuleShape(2, 1, 1), torch.device("cpu"))
        b_chan = _StubModule(Scope([0]), ModuleShape(1, 2, 1), torch.device("cpu"))
        b_rep = _StubModule(Scope([0]), ModuleShape(1, 1, 2), torch.device("cpu"))
        b_dev = _StubModule(Scope([0]), ModuleShape(1, 1, 1), torch.device("meta"))

        with pytest.raises(ScopeError):
            _validate_sum_modules([a, b_scope])  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            _validate_sum_modules([a, b_feat])  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            _validate_sum_modules([a, b_chan])  # type: ignore[arg-type]
        with pytest.raises(ShapeError):
            _validate_sum_modules([a, b_rep])  # type: ignore[arg-type]
        with pytest.raises(InvalidParameterCombinationError):
            _validate_sum_modules([a, b_dev])  # type: ignore[arg-type]

    def test_make_sum_weights_validation_error_paths(self):
        with pytest.raises(ShapeError):
            _make_sum_weights(
                weights=[[0.5, 0.5]],  # type: ignore[list-item]
                features=1,
                repetitions=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        with pytest.raises(InvalidWeightsError):
            _make_sum_weights(
                weights=[0.5, float("nan")],
                features=1,
                repetitions=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        with pytest.raises(InvalidWeightsError):
            _make_sum_weights(
                weights=[0.0, 1.0],
                features=1,
                repetitions=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        with pytest.raises(InvalidWeightsError):
            _make_sum_weights(
                weights=[65504.0, 65504.0],
                features=1,
                repetitions=1,
                device=torch.device("cpu"),
                dtype=torch.float16,
            )

    def test_dsl_context_operator_branches_and_notimplemented_paths(self):
        with dsl():
            weighted_module_mul = Normal(0) * 0.5
            weighted_module_times_weighted = Normal(0) * (0.5 * Normal(1))
            product_with_buildable = Normal(0) * term(Normal(1))
            assert isinstance(weighted_module_mul, WeightedExpr)
            assert isinstance(weighted_module_times_weighted, WeightedExpr)
            assert isinstance(product_with_buildable.build(), Module)
            assert Module.__mul__(Normal(0), "x") is NotImplemented  # type: ignore[arg-type]
            assert Module.__rmul__(Normal(0), "x") is NotImplemented  # type: ignore[arg-type]
            assert Module.__radd__(Normal(0), 1) is NotImplemented  # type: ignore[arg-type]
            with pytest.raises(InvalidParameterError):
                _ = Normal(0) + Normal(0)

    def test_dsl_context_restores_existing_module_operator_methods(self, monkeypatch):
        def _orig_mul(self: Module, other: object):  # pragma: no cover
            return ("mul", self, other)

        def _orig_rmul(self: Module, other: object):  # pragma: no cover
            return ("rmul", self, other)

        def _orig_add(self: Module, other: object):  # pragma: no cover
            return ("add", self, other)

        def _orig_radd(self: Module, other: object):  # pragma: no cover
            return ("radd", self, other)

        monkeypatch.setattr(Module, "__mul__", _orig_mul, raising=False)
        monkeypatch.setattr(Module, "__rmul__", _orig_rmul, raising=False)
        monkeypatch.setattr(Module, "__add__", _orig_add, raising=False)
        monkeypatch.setattr(Module, "__radd__", _orig_radd, raising=False)

        with dsl():
            _ = Normal(0) * Normal(1)

        assert Module.__mul__ is _orig_mul
        assert Module.__rmul__ is _orig_rmul
        assert Module.__add__ is _orig_add
        assert Module.__radd__ is _orig_radd
