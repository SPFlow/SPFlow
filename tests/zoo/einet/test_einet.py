"""Tests for Einet module."""

from itertools import product

import pytest
import torch
from torch import nn

from spflow.exceptions import InvalidParameterError, UnsupportedOperationError
from spflow.meta import Scope
from spflow.zoo.einet import Einet
from spflow.modules.leaves.normal import Normal
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


# Test parameter values
num_sums_values = [3, 8]
num_leaves_values = [3, 8]
depth_values = [0, 1, 2]
num_repetitions_values = [1, 3]
layer_type_values = ["einsum", "linsum"]
structure_values = ["top-down", "bottom-up"]

# Full parameter grid for construction/log-likelihood tests
params_full = list(
    product(
        num_sums_values,
        num_leaves_values,
        depth_values,
        num_repetitions_values,
        layer_type_values,
        structure_values,
    )
)

# Sampling only supports top-down for now
params_sampling = [p for p in params_full if p[5] == "top-down"]  # structure is 6th element (index 5)


def make_leaf_modules(num_features: int, num_leaves: int, num_repetitions: int) -> list:
    """Create leaf modules for testing."""
    return [
        Normal(
            scope=Scope([i]),
            out_channels=num_leaves,
            num_repetitions=num_repetitions,
        )
        for i in range(num_features)
    ]


class TestEinetConstruction:
    """Test Einet construction with various parameter combinations."""

    @pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", params_full)
    def test_parametrized_construction(
        self,
        num_sums: int,
        num_leaves: int,
        depth: int,
        num_reps: int,
        layer_type: str,
        structure: str,
    ):
        """Test Einet construction with various parameter combinations."""
        # Need 2^depth features for the given depth
        num_features = max(4, 2**depth)

        leaf_modules = make_leaf_modules(num_features, num_leaves, num_reps)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_reps,
            layer_type=layer_type,
            structure=structure,
        )

        assert model.num_features == num_features
        assert model.num_sums == num_sums
        assert model.num_leaves == num_leaves
        assert model.depth == depth
        assert model.num_repetitions == num_reps
        assert model.layer_type == layer_type
        assert model.structure == structure

    @pytest.mark.parametrize("num_classes", [1, 3, 5])
    def test_multi_class(self, num_classes: int):
        """Test multi-class Einet construction."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=num_classes,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
        )

        assert model.num_classes == num_classes

    def test_invalid_depth(self):
        """Test that too large depth raises error."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        with pytest.raises(ValueError):
            Einet(
                leaf_modules=leaf_modules,
                num_classes=1,
                num_sums=5,
                num_leaves=3,
                depth=10,  # Way too large for 4 features
                num_repetitions=2,
            )

    def test_invalid_layer_type(self):
        """Test that invalid layer_type raises error."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        with pytest.raises(ValueError):
            Einet(
                leaf_modules=leaf_modules,
                num_classes=1,
                num_sums=5,
                num_leaves=3,
                depth=1,
                num_repetitions=2,
                layer_type="invalid",
            )

    def test_invalid_structure(self):
        """Test that invalid structure raises error."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        with pytest.raises(ValueError):
            Einet(
                leaf_modules=leaf_modules,
                num_classes=1,
                num_sums=5,
                num_leaves=3,
                depth=1,
                num_repetitions=2,
                structure="invalid",
            )


class TestEinetLogLikelihood:
    """Test Einet log-likelihood computation."""

    @pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", params_full)
    def test_parametrized_log_likelihood(
        self,
        num_sums: int,
        num_leaves: int,
        depth: int,
        num_reps: int,
        layer_type: str,
        structure: str,
    ):
        """Test log-likelihood with various parameter combinations."""
        num_features = max(4, 2**depth)
        batch_size = 10

        leaf_modules = make_leaf_modules(num_features, num_leaves, num_reps)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=2,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_reps,
            layer_type=layer_type,
            structure=structure,
        )

        data = torch.randn(batch_size, num_features)
        lls = model.log_likelihood(data)

        assert lls.shape[0] == batch_size
        assert torch.isfinite(lls).all()

    def test_log_likelihood_cached(self):
        """Test log-likelihood caching."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
        )

        data = torch.randn(10, 4)
        cache = Cache()
        lls = model.log_likelihood(data, cache=cache)

        assert "log_likelihood" in cache
        assert torch.isfinite(lls).all()


class TestEinetSampling:
    """Test Einet sampling."""

    @pytest.mark.parametrize("num_sums,num_leaves,depth,num_reps,layer_type,structure", params_sampling)
    def test_parametrized_sampling(
        self,
        num_sums: int,
        num_leaves: int,
        depth: int,
        num_reps: int,
        layer_type: str,
        structure: str,
    ):
        """Test sampling with various parameter combinations."""
        num_features = max(4, 2**depth)
        num_samples = 20

        leaf_modules = make_leaf_modules(num_features, num_leaves, num_reps)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=num_sums,
            num_leaves=num_leaves,
            depth=depth,
            num_repetitions=num_reps,
            layer_type=layer_type,
            structure=structure,
        )

        samples = model.sample(num_samples=num_samples)

        assert samples.shape == (num_samples, num_features)
        assert torch.isfinite(samples).all()

    @pytest.mark.parametrize("layer_type", layer_type_values)
    def test_mpe_sampling(self, layer_type: str):
        """Test MPE sampling (top-down only)."""
        num_features = 4
        num_samples = 10
        leaf_modules = make_leaf_modules(num_features, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            layer_type=layer_type,
            structure="top-down",
        )

        samples = model.sample(num_samples=num_samples, is_mpe=True)

        assert samples.shape == (num_samples, num_features)
        assert torch.isfinite(samples).all()

    def test_bottom_up_sampling_not_implemented(self):
        """Test that bottom-up sampling raises NotImplementedError."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            structure="bottom-up",
        )

        with pytest.raises(NotImplementedError):
            model.sample(num_samples=10)


class TestEinetGradient:
    """Test gradient flow through Einet."""

    @pytest.mark.parametrize("layer_type,structure", product(layer_type_values, structure_values))
    def test_gradient_flow(self, layer_type: str, structure: str):
        """Test that gradients flow through log-likelihood."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            layer_type=layer_type,
            structure=structure,
        )

        data = torch.randn(20, 4)
        lls = model.log_likelihood(data)
        loss = -lls.mean()
        loss.backward()

        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("layer_type,structure", product(layer_type_values, structure_values))
    def test_optimization(self, layer_type: str, structure: str):
        """Test that optimization updates parameters."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=1,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            layer_type=layer_type,
            structure=structure,
        )

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        data = torch.randn(50, 4)

        for _ in range(5):
            optimizer.zero_grad()
            lls = model.log_likelihood(data)
            loss = -lls.mean()
            loss.backward()
            optimizer.step()

        # Check that some parameters changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], rtol=0.0, atol=0.0):
                changed = True
                break

        assert changed, "No parameters changed during optimization"


class TestEinetExtraRepr:
    """Test string representation."""

    @pytest.mark.parametrize("layer_type,structure", product(layer_type_values, structure_values))
    def test_extra_repr(self, layer_type: str, structure: str):
        """Test extra_repr includes configuration."""
        leaf_modules = make_leaf_modules(4, 3, 2)
        model = Einet(
            leaf_modules=leaf_modules,
            num_classes=2,
            num_sums=5,
            num_leaves=3,
            depth=1,
            num_repetitions=2,
            layer_type=layer_type,
            structure=structure,
        )

        repr_str = model.extra_repr()
        assert "num_features=4" in repr_str
        assert "num_classes=2" in repr_str
        assert f"layer_type={layer_type}" in repr_str
        assert f"structure={structure}" in repr_str


class TestEinetAdditionalCoverage:
    """Targeted tests for less common branches and delegating methods."""

    def test_more_invalid_constructor_parameters(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        with pytest.raises(ValueError):
            Einet(leaf_modules=leaf_modules, num_classes=0)
        with pytest.raises(ValueError):
            Einet(leaf_modules=leaf_modules, num_sums=0)
        with pytest.raises(ValueError):
            Einet(leaf_modules=leaf_modules, num_leaves=0)
        with pytest.raises(ValueError):
            Einet(leaf_modules=leaf_modules, depth=-1)
        with pytest.raises(ValueError):
            Einet(leaf_modules=leaf_modules, num_repetitions=0)

    def test_properties_delegate_to_root(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=1, num_repetitions=1)
        assert model.n_out == 1
        assert model.feature_to_scope.shape == model.root_node.feature_to_scope.shape
        with pytest.raises(AttributeError):
            _ = type(model).scopes_out.fget(model)

    def test_log_posterior_raises_for_single_class(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=1, num_repetitions=1)
        data = torch.randn(3, 4)
        with pytest.raises(UnsupportedOperationError):
            model.log_posterior(data)

    def test_log_posterior_and_predict_proba_multiclass(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=3, depth=1, num_repetitions=1)
        data = torch.randn(5, 4)

        log_post = model.log_posterior(data)
        proba = model.predict_proba(data)

        assert log_post.shape == (5, 3)
        assert proba.shape == (5, 3)
        assert torch.isfinite(log_post).all()
        assert torch.allclose(proba.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_multiclass_sample_mpe_and_stochastic(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=3, depth=1, num_repetitions=1)
        data = torch.full((6, 4), torch.nan)

        class DummyInput(nn.Module):
            def __init__(self, num_features: int):
                super().__init__()
                self.num_features = num_features

            def _sample(self, data, sampling_ctx, is_mpe, cache):
                del sampling_ctx
                return torch.zeros((data.shape[0], self.num_features))

        class DummyRoot(nn.Module):
            def __init__(self, num_classes: int, num_features: int):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros((1, num_classes, 1)))
                self.inputs = DummyInput(num_features)

        model.root_node = DummyRoot(model.num_classes, model.num_features)

        mpe_samples = model.sample(data=data, is_mpe=True)
        random_samples = model.sample(data=data, is_mpe=False)

        assert mpe_samples.shape == (6, 4)
        assert random_samples.shape == (6, 4)
        assert torch.isfinite(mpe_samples).all()
        assert torch.isfinite(random_samples).all()

    def test_multiclass_sampling_raises_for_invalid_logits_shape(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=3, depth=1, num_repetitions=1)
        model.root_node.logits = torch.nn.Parameter(torch.zeros(2, 2))

        with pytest.raises(InvalidParameterError):
            model.sample(num_samples=2)

    def test_sample_defaults_to_one_sample(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=1, num_repetitions=1)
        samples = model.sample()
        assert samples.shape == (1, 4)

    def test_sample_initializes_repetition_index_for_single_rep(self):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=1, num_repetitions=1)
        sampling_ctx = SamplingContext(num_samples=4)
        data = torch.full((4, 4), torch.nan)
        cache = Cache()

        samples = model._sample(data=data, sampling_ctx=sampling_ctx, cache=cache)

        assert samples.shape == (4, 4)
        assert sampling_ctx.repetition_idx is not None
        assert torch.equal(sampling_ctx.repetition_idx, torch.zeros(4, dtype=torch.long))

    def test_delegating_methods(self, monkeypatch: pytest.MonkeyPatch):
        leaf_modules = make_leaf_modules(4, 3, 1)
        model = Einet(leaf_modules=leaf_modules, num_classes=1, num_repetitions=1)
        data = torch.randn(3, 4)
        cache = Cache()

        calls = {"em": 0, "marginalize": 0}
        expected_result = object()

        def fake_em(call_data, bias_correction=True, *, cache=None):
            assert call_data is data
            assert bias_correction is True
            assert cache is cache_obj
            calls["em"] += 1

        def fake_marginalize(marg_rvs, prune=True, cache=None):
            assert marg_rvs == [0, 2]
            assert prune is False
            assert cache is cache_obj
            calls["marginalize"] += 1
            return expected_result

        cache_obj = cache
        monkeypatch.setattr(model.root_node, "_expectation_maximization_step", fake_em)
        monkeypatch.setattr(model.root_node, "marginalize", fake_marginalize)

        model._expectation_maximization_step(data, cache=cache_obj)
        with pytest.raises(AttributeError):
            model.maximum_likelihood_estimation(data)
        result = model.marginalize([0, 2], prune=False, cache=cache_obj)

        assert calls["em"] == 1
        assert calls["marginalize"] == 1
        assert result is expected_result
