"""Unit tests for the example-oriented DSL construction helpers."""

import pytest
import torch

from spflow.dsl import dsl, term
from spflow.exceptions import InvalidParameterError, ScopeError, ShapeError
from spflow.modules.leaves.normal import Normal
from spflow.modules.module import Module
from spflow.modules.sums.sum import Sum


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
