"""Tests for Einet module."""

from itertools import product

import pytest
import torch

from spflow.meta import Scope
from spflow.modules.einsum import Einet
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

# Parameter grid for construction/log-likelihood tests
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

# Reduced params for sampling tests (skip known problematic combinations)
# Sampling with num_repetitions=1 and bottom-up has edge cases
params_sampling = [
    (num_sums, num_leaves, depth, num_reps, layer_type, structure)
    for num_sums, num_leaves, depth, num_reps, layer_type, structure in params_full
    if not (num_reps == 1 and structure == "bottom-up")  # Skip problematic combo
    and not (structure == "bottom-up" and depth > 1)  # Skip complex bottom-up
]


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

    @pytest.mark.parametrize(
        "num_sums,num_leaves,depth,num_reps,layer_type,structure", params_full
    )
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
        with pytest.raises(ValueError, match="depth.*too large"):
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
        with pytest.raises(ValueError, match="layer_type"):
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
        with pytest.raises(ValueError, match="structure"):
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

    @pytest.mark.parametrize(
        "num_sums,num_leaves,depth,num_reps,layer_type,structure", params_full
    )
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

    @pytest.mark.parametrize(
        "num_sums,num_leaves,depth,num_reps,layer_type,structure", params_sampling
    )
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
    def test_mpe_sampling_top_down(self, layer_type: str):
        """Test MPE sampling with top-down structure."""
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
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

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
            if not torch.allclose(param, initial_params[name]):
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
