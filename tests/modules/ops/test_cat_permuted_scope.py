"""Test Cat.sample with permuted scopes."""

import pytest
import torch
import numpy as np

from spflow.meta import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.modules.ops.cat import Cat
from spflow.utils.cache import Cache
from spflow.utils.sampling_context import SamplingContext


class IndexLeaf(LeafModule):
    """Deterministic leaf that outputs feature index for each query variable.

    For testing scope order invariance: if scope.query = [7, 2, 5, 4],
    then samples[:, 0] = 7, samples[:, 1] = 2, samples[:, 2] = 5, samples[:, 3] = 4.

    This allows us to easily verify that samples are placed at the correct
    data positions regardless of scope order.
    """

    def __init__(
        self,
        scope: Scope | int | list[int],
        out_channels: int = 1,
        num_repetitions: int = 1,
    ):
        # Create dummy params matching expected shapes
        num_features = len(scope) if isinstance(scope, (list, tuple)) else 1
        if isinstance(scope, Scope):
            num_features = len(scope.query)
        dummy_param = torch.zeros(num_features, out_channels, num_repetitions)
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[dummy_param],
        )

    @property
    def _torch_distribution_class(self):
        return torch.distributions.Normal

    @property
    def _supported_value(self) -> float:
        return 0.0

    def params(self):
        # Return dummy params - not used for sampling
        return {"loc": torch.zeros(1), "scale": torch.ones(1)}

    def _compute_parameter_estimates(self, data, weights, bias_correction):
        return {}

    @property
    def mode(self):
        # Mode returns the scope indices themselves
        indices = torch.tensor(self.scope.query, dtype=torch.float32)
        # Shape: [out_features, out_channels, num_repetitions]
        return indices.view(-1, 1, 1).expand(-1, self.out_shape.channels, self.out_shape.repetitions)

    def _sample(
        self,
        data: torch.Tensor,
        sampling_ctx: SamplingContext,
        cache: Cache,
        is_mpe: bool = False,
    ) -> torch.Tensor:
        """Return deterministic values based on scope indices."""
        del cache
        del is_mpe

        # Find which positions need sampling (NaN and in scope)
        out_of_scope = [x for x in range(data.shape[1]) if x not in self.scope.query]
        marg_mask = torch.isnan(data)
        marg_mask[:, out_of_scope] = False

        samples_mask = marg_mask
        samples_mask[:, self.scope.query] &= sampling_ctx.mask

        instance_mask = samples_mask.sum(1) > 0
        n_samples = instance_mask.sum()

        if n_samples == 0:
            return data

        # Place scope indices at the correct positions
        # For each query variable in our scope, put its index value there
        for _, rv_idx in enumerate(self.scope.query):
            # Only fill where the mask is True for this position
            mask_for_rv = samples_mask[:, rv_idx]
            data[mask_for_rv, rv_idx] = float(rv_idx)

        return data


class TestCatPermutedScope:
    """Test that Cat.sample works correctly with permuted scopes."""

    def test_cat_sequential_scope(self):
        """Test Cat.sample with sequential scope (baseline)."""
        num_samples = 3
        num_features = 8
        out_channels = 1
        num_repetitions = 1

        # Sequential scopes
        scope1 = [0, 1, 2, 3]
        scope2 = [4, 5, 6, 7]

        leaf1 = IndexLeaf(scope=scope1, out_channels=out_channels, num_repetitions=num_repetitions)
        leaf2 = IndexLeaf(scope=scope2, out_channels=out_channels, num_repetitions=num_repetitions)

        cat = Cat(inputs=[leaf1, leaf2], dim=1)

        # Sample
        data = torch.full((num_samples, num_features), float("nan"))
        channel_index = torch.zeros(num_samples, num_features, dtype=torch.long)
        mask = torch.ones(num_samples, num_features, dtype=torch.bool)
        repetition_idx = torch.zeros(num_samples, dtype=torch.long)

        sampling_ctx = SamplingContext(
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_idx,
        )

        samples = cat.sample(data=data)

        # Expected: each position i should have value i
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * num_samples)

        assert torch.allclose(samples, expected, rtol=0.0, atol=0.0), f"Expected {expected}, got {samples}"

    def test_cat_permuted_scope(self):
        """Test Cat.sample with permuted scope - the critical test case."""
        num_samples = 3
        num_features = 8
        out_channels = 1
        num_repetitions = 1

        # Permuted scopes (like random evidence/query split)
        scope1 = [3, 1, 6, 0]  # Evidence-like (permuted)
        scope2 = [7, 2, 5, 4]  # Query-like (permuted)

        leaf1 = IndexLeaf(scope=scope1, out_channels=out_channels, num_repetitions=num_repetitions)
        leaf2 = IndexLeaf(scope=scope2, out_channels=out_channels, num_repetitions=num_repetitions)

        cat = Cat(inputs=[leaf1, leaf2], dim=1)

        # Sample
        data = torch.full((num_samples, num_features), float("nan"))
        channel_index = torch.zeros(num_samples, num_features, dtype=torch.long)
        mask = torch.ones(num_samples, num_features, dtype=torch.bool)
        repetition_idx = torch.zeros(num_samples, dtype=torch.long)

        sampling_ctx = SamplingContext(
            channel_index=channel_index,
            mask=mask,
            repetition_index=repetition_idx,
        )

        samples = cat.sample(data=data)

        # Expected: each position i should have value i (the RV index)
        # Position 0 -> value 0, Position 1 -> value 1, etc.
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]] * num_samples)

        assert torch.allclose(samples, expected, rtol=0.0, atol=0.0), (
            f"Expected each position i to have value i.\n"
            f"Expected: {expected[0].tolist()}\n"
            f"Got: {samples[0].tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
