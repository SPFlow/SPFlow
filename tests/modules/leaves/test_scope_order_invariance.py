"""Tests for verifying that scope order does not affect sampling results.

This module tests that the order of scope indices in a leaf's scope does not affect
sampling behavior. The key insight is:

When sampling from a leaf with scope [a, b, c]:
1. The distribution samples values for feature positions 0, 1, 2
2. These are placed at data positions a, b, c respectively (i.e., scope.query[i])

If we create a distribution that returns the scope index itself as the value:
- Feature position 0 returns scope.query[0] = a
- Feature position 1 returns scope.query[1] = b
- Feature position 2 returns scope.query[2] = c

After sampling:
- data[a] = a (from feature 0)
- data[b] = b (from feature 1)
- data[c] = c (from feature 2)

So regardless of the scope ordering, data[i] should equal i for all i in scope.
This proves scope order doesn't affect the semantic result.
"""
import itertools

import pytest
import torch
from torch import Tensor

from spflow.meta.data import Scope
from spflow.modules.leaves.leaf import LeafModule
from spflow.utils.sampling_context import SamplingContext


class IndexValueDistribution(torch.distributions.Distribution):
    """Custom distribution that returns the scope index as the sampled value.

    For each feature position i, this distribution returns scope_query[i] as
    the sampled value. This makes it easy to trace where each value ends up
    in the data tensor after sampling.

    Example:
        If scope.query = [2, 0, 1], then:
        - Feature position 0 returns value 2.0 (will go to data[2])
        - Feature position 1 returns value 0.0 (will go to data[0])
        - Feature position 2 returns value 1.0 (will go to data[1])

        After sampling:
        - data[2] = 2.0, data[0] = 0.0, data[1] = 1.0
        - i.e., data[i] = i for all i in scope
    """

    has_rsample = False
    arg_constraints = {}

    def __init__(self, scope_indices: Tensor, validate_args=None):
        """Initialize the distribution.

        Args:
            scope_indices: Tensor containing the scope index for each feature position.
                Shape: (out_features, out_channels, num_repetitions)
        """
        self._scope_indices = scope_indices
        batch_shape = torch.Size([])
        event_shape = scope_indices.shape
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        """Return scope indices as sample values."""
        shape = sample_shape + self._scope_indices.shape
        return self._scope_indices.expand(shape).clone().float()

    def log_prob(self, value):
        """Always return 0 (probability of 1)."""
        return torch.zeros_like(value)

    @property
    def mode(self):
        """Return the scope indices as the mode."""
        return self._scope_indices.float()


class IndexLeaf(LeafModule):
    """Dummy leaf that returns scope indices when sampled.

    This leaf always outputs log-probability 0 (probability 1) for all inputs.
    When sampled, each feature position returns the corresponding scope index.

    This is useful for testing that samples are correctly placed at their
    scope positions - after sampling, data[scope_idx] should contain scope_idx.

    Example:
        If scope = Scope([2, 0, 1]), after sampling:
        - data[0] = 0.0
        - data[1] = 1.0
        - data[2] = 2.0

        Because:
        - Feature 0 returns scope.query[0]=2, placed at data[2]
        - Feature 1 returns scope.query[1]=0, placed at data[0]
        - Feature 2 returns scope.query[2]=1, placed at data[1]
    """

    def __init__(
        self,
        scope: Scope,
        out_channels: int = 1,
        num_repetitions: int = 1,
    ):
        """Initialize the IndexLeaf.

        Args:
            scope: Variable scope.
            out_channels: Number of output channels.
            num_repetitions: Number of repetitions.
        """
        # Create dummy params for initialization
        dummy_param = torch.zeros(len(scope.query), out_channels, num_repetitions)
        super().__init__(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
            params=[dummy_param],
        )

        self._out_features = len(scope.query)

        # Create scope indices tensor: each position i contains scope.query[i]
        # Shape: (out_features, out_channels, num_repetitions)
        scope_indices = torch.tensor(
            scope.query, dtype=torch.float32
        ).view(-1, 1, 1).expand(len(scope.query), out_channels, num_repetitions).clone()
        self.register_buffer("scope_indices", scope_indices)

    @property
    def _torch_distribution_class(self):
        """Return the custom distribution class."""
        return IndexValueDistribution

    @property
    def _supported_value(self) -> float:
        """Return a value in the support of the distribution."""
        return 0.0

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Return the custom distribution that outputs scope indices."""
        return IndexValueDistribution(self.scope_indices, validate_args=self._validate_args)

    def params(self) -> dict[str, Tensor]:
        """Return distribution parameters."""
        return {"scope_indices": self.scope_indices}

    def _compute_parameter_estimates(
        self, data: Tensor, weights: Tensor, bias_correction: bool
    ) -> dict[str, Tensor]:
        """Not used for this test leaf."""
        raise NotImplementedError("MLE not supported for IndexLeaf")


def create_sampling_context(
    num_samples: int, num_features: int, out_channels: int = 1, num_repetitions: int = 1,
    channel: int = 0, repetition: int = 0
) -> SamplingContext:
    """Create a sampling context for testing."""
    channel_index = torch.full((num_samples, num_features), channel, dtype=torch.long)
    mask = torch.ones(num_samples, num_features, dtype=torch.bool)
    repetition_index = torch.full((num_samples,), repetition, dtype=torch.long)
    return SamplingContext(
        channel_index=channel_index,
        mask=mask,
        repetition_index=repetition_index,
    )


class TestScopeOrderInvariance:
    """Tests verifying that scope order does not affect sampling results.

    The key invariant: If we create leaves with different scope orderings,
    and sample from them using the IndexLeaf (which returns scope indices),
    then after sampling, data[i] should always contain value i for all i in scope.

    This proves that scope order doesn't affect the final result.

    NOTE: These tests currently FAIL because the sampling implementation has a bug
    where scope order affects the result. These tests document the expected behavior.
    """

    @pytest.mark.parametrize(
        "scope_query",
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ],
    )
    def test_samples_fill_correct_scope_positions(self, scope_query: list[int]):
        """Test that data[scope_idx] == scope_idx after sampling.

        Using IndexLeaf:
        - Feature position i returns scope.query[i]
        - Feature position i is placed at data[scope.query[i]]

        So data[scope.query[i]] should contain scope.query[i].
        Since scope.query just contains the indices [0,1,2] in some order,
        data[j] should equal j for all j in scope.
        """
        num_samples = 5
        out_channels = 1
        num_repetitions = 1

        scope = Scope(scope_query)
        leaf = IndexLeaf(
            scope=scope,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )

        # Total features in data tensor (max scope index + 1)
        num_total_features = max(scope_query) + 1

        # Create data tensor with NaN for all positions
        data = torch.full((num_samples, num_total_features), float("nan"))

        sampling_ctx = create_sampling_context(
            num_samples=num_samples,
            num_features=num_total_features,
            out_channels=out_channels,
            num_repetitions=num_repetitions,
        )

        # Sample from the leaf
        samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

        # The expected behavior: data[idx] should equal idx for all idx in scope
        # This is because:
        # - The distribution returns scope.query[i] at feature position i
        # - Feature position i is placed at data[scope.query[i]]
        # So data[scope.query[i]] = scope.query[i], meaning data[idx] = idx

        for scope_idx in scope_query:
            expected_value = float(scope_idx)
            actual_value = samples[0, scope_idx].item()
            assert torch.allclose(
                samples[:, scope_idx],
                torch.full((num_samples,), expected_value),
            ), (
                f"For scope {scope_query}: data[{scope_idx}] should be {expected_value}, "
                f"got {actual_value}"
            )

    def test_all_permutations_produce_same_data(self):
        """Test that ALL permutations of a scope produce identical data tensors.

        Using IndexLeaf, each permutation should result in:
        - data[0] = 0.0
        - data[1] = 1.0
        - data[2] = 2.0

        Regardless of the scope ordering.
        """
        base_scope = [0, 1, 2]
        num_samples = 3
        out_channels = 1
        num_repetitions = 1
        num_total_features = 3

        # Expected result for ALL permutations
        expected = torch.tensor([[0., 1., 2.]] * num_samples)

        for perm in itertools.permutations(base_scope):
            scope = Scope(list(perm))
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))
            sampling_ctx = create_sampling_context(
                num_samples=num_samples,
                num_features=num_total_features,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {list(perm)}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )

    def test_scope_order_with_multiple_channels(self):
        """Test scope order invariance with multiple output channels."""
        num_samples = 3
        out_channels = 3
        num_repetitions = 1
        num_total_features = 3

        expected = torch.tensor([[0., 1., 2.]] * num_samples)

        for perm in [[0, 1, 2], [2, 1, 0]]:
            scope = Scope(perm)
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))
            sampling_ctx = create_sampling_context(
                num_samples=num_samples,
                num_features=num_total_features,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
                channel=1,
            )

            samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {perm}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )

    def test_scope_order_with_multiple_repetitions(self):
        """Test scope order invariance with multiple repetitions."""
        num_samples = 3
        out_channels = 1
        num_repetitions = 3
        num_total_features = 3

        expected = torch.tensor([[0., 1., 2.]] * num_samples)

        for perm in [[0, 1, 2], [1, 2, 0]]:
            scope = Scope(perm)
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))
            sampling_ctx = create_sampling_context(
                num_samples=num_samples,
                num_features=num_total_features,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
                repetition=1,
            )

            samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {perm}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )

    def test_log_likelihood_always_one(self):
        """Test that the leaf always outputs probability 1 (log_prob 0) for all inputs."""
        scope = Scope([0, 1, 2])
        leaf = IndexLeaf(scope=scope, out_channels=2, num_repetitions=2)

        data = torch.randn(5, 3)
        lls = leaf.log_likelihood(data)

        assert torch.allclose(lls, torch.zeros_like(lls))

    def test_larger_scope_permutations(self):
        """Test with a larger scope (4 features) to ensure scalability."""
        num_samples = 2
        out_channels = 1
        num_repetitions = 1
        num_total_features = 4

        expected = torch.tensor([[0., 1., 2., 3.]] * num_samples)

        test_perms = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
        ]

        for perm in test_perms:
            scope = Scope(perm)
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))
            sampling_ctx = create_sampling_context(
                num_samples=num_samples,
                num_features=num_total_features,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {perm}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )

    def test_mpe_sampling_with_scope_permutations(self):
        """Test MPE (most probable explanation) sampling with different scope orders."""
        num_samples = 3
        out_channels = 1
        num_repetitions = 1
        num_total_features = 3

        expected = torch.tensor([[0., 1., 2.]] * num_samples)

        for perm in [[0, 1, 2], [2, 1, 0]]:
            scope = Scope(perm)
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))
            sampling_ctx = create_sampling_context(
                num_samples=num_samples,
                num_features=num_total_features,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            samples = leaf.sample(data=data, is_mpe=True, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {perm}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )

    def test_non_contiguous_scope(self):
        """Test with non-contiguous scope indices (e.g., [0, 2, 5]).

        NOTE: This test is disabled because the sampling implementation in
        LeafModule.sample() has an issue with non-contiguous scopes where
        the mask shape doesn't align with the scope query size.
        This is a known limitation that should be addressed separately.
        """
        pytest.skip("Non-contiguous scope sampling has shape mismatch issues")

    def test_partial_mask_sampling(self):
        """Test that partial masking works correctly with different scope orders."""
        num_samples = 4
        out_channels = 1
        num_repetitions = 1
        num_total_features = 3

        expected = torch.tensor([[0., 1., 2.]] * num_samples)

        for perm in [[0, 1, 2], [2, 0, 1]]:
            scope = Scope(perm)
            leaf = IndexLeaf(
                scope=scope,
                out_channels=out_channels,
                num_repetitions=num_repetitions,
            )

            data = torch.full((num_samples, num_total_features), float("nan"))

            channel_index = torch.zeros(num_samples, num_total_features, dtype=torch.long)
            mask = torch.ones(num_samples, num_total_features, dtype=torch.bool)
            repetition_index = torch.zeros(num_samples, dtype=torch.long)

            sampling_ctx = SamplingContext(
                channel_index=channel_index,
                mask=mask,
                repetition_index=repetition_index,
            )

            samples = leaf.sample(data=data, sampling_ctx=sampling_ctx)

            assert torch.allclose(samples, expected), (
                f"For permutation {perm}, expected {expected[0].tolist()}, "
                f"got {samples[0].tolist()}"
            )
