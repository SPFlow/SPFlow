"""Tests for FeatureContext class."""

import pytest

from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import (
    BernoulliType,
    ExponentialType,
    NormalType,
    PoissonType,
)
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope


class TestFeatureContextInitialization:
    """Test FeatureContext initialization and basic functionality."""

    def test_feature_context_init_dict(self):
        """Test FeatureContext with dict initialization."""
        scope = Scope([0, 1])
        domains = {0: NormalType(), 1: BernoulliType()}
        ctx = FeatureContext(scope, domains)

        assert ctx.scope.query == [0, 1]
        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)

    def test_feature_context_init_list(self):
        """Test FeatureContext with list initialization."""
        scope = Scope([0, 1, 2])
        domains = [NormalType(), BernoulliType(), PoissonType()]
        ctx = FeatureContext(scope, domains)

        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)
        assert isinstance(ctx.domain_map[2], PoissonType)

    def test_feature_context_init_none(self):
        """Test FeatureContext with None domains (defaults to Unknown)."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, None)

        assert ctx.domain_map[0] == MetaType.Unknown
        assert ctx.domain_map[1] == MetaType.Unknown

    def test_feature_context_init_empty_dict(self):
        """Test FeatureContext with empty dict (defaults to Unknown)."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, {})

        assert ctx.domain_map[0] == MetaType.Unknown
        assert ctx.domain_map[1] == MetaType.Unknown

    def test_feature_context_init_scope(self):
        """Test FeatureContext stores scope correctly."""
        scope = Scope([0, 1, 2], evidence=[3])
        ctx = FeatureContext(scope)

        assert ctx.scope.query == [0, 1, 2]
        assert ctx.scope.evidence == [3]


class TestFeatureContextParseType:
    """Test parse_type class method."""

    def test_parse_type_feature_type_instance(self):
        """Test parse_type with FeatureType instance."""
        instance = NormalType(mean=5.0, std=2.0)
        result = FeatureContext.parse_type(instance)

        assert result is instance
        assert result.mean == 5.0
        assert result.std == 2.0

    def test_parse_type_feature_type_class(self):
        """Test parse_type with FeatureType class."""
        result = FeatureContext.parse_type(NormalType)

        assert isinstance(result, NormalType)
        assert result.mean == 0.0  # default value
        assert result.std == 1.0  # default value

    def test_parse_type_categorical(self):
        """Test parse_type with BernoulliType class."""
        result = FeatureContext.parse_type(BernoulliType)

        assert isinstance(result, BernoulliType)

    def test_parse_type_meta_type(self):
        """Test parse_type with MetaType (already not a class)."""
        result = FeatureContext.parse_type(MetaType.Continuous)

        assert result == MetaType.Continuous


class TestFeatureContextSetDomains:
    """Test set_domains method."""

    def test_set_domains_dict(self):
        """Test set_domains with dict input."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        domains = {0: NormalType(), 1: BernoulliType(), 2: PoissonType()}
        ctx.set_domains(domains)

        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)
        assert isinstance(ctx.domain_map[2], PoissonType)

    def test_set_domains_list(self):
        """Test set_domains with list input."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        domains = [NormalType(), BernoulliType(), PoissonType()]
        ctx.set_domains(domains)

        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)
        assert isinstance(ctx.domain_map[2], PoissonType)

    def test_set_domains_with_classes(self):
        """Test set_domains with FeatureType classes (not instances)."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)

        ctx.set_domains({0: NormalType, 1: BernoulliType})

        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)

    def test_set_domains_with_overwrite_true(self):
        """Test set_domains overwriting existing domains."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType()})

        # Overwrite with new domains
        ctx.set_domains({0: PoissonType(), 1: BernoulliType()}, overwrite=True)

        assert isinstance(ctx.domain_map[0], PoissonType)
        assert isinstance(ctx.domain_map[1], BernoulliType)

    def test_set_domains_overwrite_false_error(self):
        """Test set_domains raises error if no overwrite and domains exist."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, {0: NormalType()})

        # Try to overwrite without permission
        with pytest.raises(ValueError, match="already specified"):
            ctx.set_domains({0: PoissonType()}, overwrite=False)

    def test_set_domains_overwrite_unknown_is_allowed(self):
        """Test set_domains allows overwriting MetaType.Unknown without overwrite flag."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)  # Defaults to Unknown

        # Should succeed without overwrite=True since default is Unknown
        ctx.set_domains({0: NormalType()}, overwrite=False)
        assert isinstance(ctx.domain_map[0], NormalType)


class TestFeatureContextSetDomainsErrors:
    """Test error handling in set_domains."""

    def test_set_domains_invalid_feature_id(self):
        """Test set_domains with invalid feature_id."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)

        # Feature 5 doesn't exist in scope
        with pytest.raises(ValueError, match="not part of the query scope"):
            ctx.set_domains({5: NormalType()})

    def test_set_domains_list_shape_mismatch(self):
        """Test set_domains with wrong number of domains."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        # Only 2 domains for 3 features
        with pytest.raises(ValueError, match="does not match number of scope query"):
            ctx.set_domains([NormalType(), BernoulliType()])

    def test_set_domains_list_too_many(self):
        """Test set_domains with too many domains."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)

        # 3 domains for 2 features
        with pytest.raises(ValueError, match="does not match number of scope query"):
            ctx.set_domains([NormalType(), BernoulliType(), PoissonType()])


class TestFeatureContextGetDomains:
    """Test get_domains method."""

    def test_get_domains_all(self):
        """Test get_domains() with no parameters (all features)."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType()})

        result = ctx.get_domains()

        assert len(result) == 3
        assert isinstance(result[0], NormalType)
        assert isinstance(result[1], BernoulliType)
        assert isinstance(result[2], PoissonType)

    def test_get_domains_single_feature_int(self):
        """Test get_domains with single int feature_id."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType()})

        result = ctx.get_domains(1)

        assert isinstance(result, BernoulliType)

    def test_get_domains_specific_features(self):
        """Test get_domains with specific feature_ids."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType()})

        result = ctx.get_domains([0, 2])

        assert len(result) == 2
        assert isinstance(result[0], NormalType)
        assert isinstance(result[1], PoissonType)

    def test_get_domains_none_parameter(self):
        """Test get_domains with None parameter (synonym for all)."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType()})

        result = ctx.get_domains(None)

        assert len(result) == 2
        assert isinstance(result[0], NormalType)
        assert isinstance(result[1], BernoulliType)

    def test_get_domains_no_domains_set(self):
        """Test get_domains when domains not yet set (returns MetaType.Unknown)."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)

        result = ctx.get_domains()

        assert result[0] == MetaType.Unknown
        assert result[1] == MetaType.Unknown


class TestFeatureContextSelect:
    """Test select method."""

    def test_select_subset_features(self):
        """Test select() creates subset context."""
        scope = Scope([0, 1, 2, 3])
        ctx = FeatureContext(
            scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType(), 3: BernoulliType()}
        )

        # Select features 1 and 3
        subset_ctx = ctx.select([1, 3])

        assert subset_ctx.scope.query == [1, 3]
        assert len(subset_ctx.domain_map) == 2
        assert isinstance(subset_ctx.domain_map[1], BernoulliType)
        assert isinstance(subset_ctx.domain_map[3], BernoulliType)

    def test_select_single_feature_int(self):
        """Test select() with single int (converted to list)."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType()})

        subset_ctx = ctx.select(1)

        assert subset_ctx.scope.query == [1]
        assert isinstance(subset_ctx.domain_map[1], BernoulliType)

    def test_select_preserves_types(self):
        """Test select() preserves feature types."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: NormalType(mean=10.0, std=5.0), 1: BernoulliType(), 2: PoissonType()})

        subset_ctx = ctx.select([0])

        assert isinstance(subset_ctx.domain_map[0], NormalType)
        assert subset_ctx.domain_map[0].mean == 10.0
        assert subset_ctx.domain_map[0].std == 5.0

    def test_select_preserves_evidence(self):
        """Test select() preserves evidence scope."""
        scope = Scope([0, 1, 2], evidence=[5, 6])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType()})

        subset_ctx = ctx.select([0, 2])

        assert subset_ctx.scope.evidence == [5, 6]

    def test_select_all_features(self):
        """Test select() with all features."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope, {0: NormalType(), 1: BernoulliType()})

        subset_ctx = ctx.select([0, 1])

        assert subset_ctx.scope.query == [0, 1]
        assert isinstance(subset_ctx.domain_map[0], NormalType)
        assert isinstance(subset_ctx.domain_map[1], BernoulliType)


class TestFeatureContextEdgeCases:
    """Test edge cases in FeatureContext."""

    def test_feature_context_single_feature(self):
        """Test FeatureContext with single feature."""
        scope = Scope(0)
        ctx = FeatureContext(scope, {0: NormalType()})

        assert len(ctx.domain_map) == 1
        assert isinstance(ctx.domain_map[0], NormalType)

    def test_feature_context_many_features(self):
        """Test FeatureContext with 100+ features."""
        n_features = 150
        scope = Scope(list(range(n_features)))

        # Create domains for all features
        domains = {i: NormalType() if i % 2 == 0 else BernoulliType() for i in range(n_features)}
        ctx = FeatureContext(scope, domains)

        assert len(ctx.domain_map) == n_features
        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)
        assert isinstance(ctx.domain_map[148], NormalType)
        assert isinstance(ctx.domain_map[149], BernoulliType)

    def test_feature_context_non_contiguous_features(self):
        """Test FeatureContext with non-contiguous feature indices."""
        scope = Scope([0, 5, 10, 100])
        ctx = FeatureContext(
            scope, {0: NormalType(), 5: BernoulliType(), 10: PoissonType(), 100: BernoulliType()}
        )

        assert len(ctx.domain_map) == 4
        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[5], BernoulliType)
        assert isinstance(ctx.domain_map[10], PoissonType)
        assert isinstance(ctx.domain_map[100], BernoulliType)


class TestFeatureContextMixedTypes:
    """Test FeatureContext with different feature types."""

    def test_feature_context_all_continuous(self):
        """Test context with all continuous types."""
        scope = Scope([0, 1, 2, 3])
        ctx = FeatureContext(
            scope, {0: NormalType(), 1: ExponentialType(), 2: NormalType(), 3: ExponentialType()}
        )

        # All should be continuous
        for i in range(4):
            assert ctx.domain_map[i].meta_type == MetaType.Continuous

    def test_feature_context_all_discrete(self):
        """Test context with all discrete types."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope, {0: BernoulliType(), 1: PoissonType(), 2: BernoulliType()})

        # All should be discrete
        for i in range(3):
            assert ctx.domain_map[i].meta_type == MetaType.Discrete

    def test_feature_context_mixed_types(self):
        """Test context with different feature types."""
        scope = Scope([0, 1, 2, 3])
        ctx = FeatureContext(
            scope, {0: NormalType(), 1: BernoulliType(), 2: PoissonType(), 3: ExponentialType()}
        )

        assert ctx.domain_map[0].meta_type == MetaType.Continuous
        assert ctx.domain_map[1].meta_type == MetaType.Discrete
        assert ctx.domain_map[2].meta_type == MetaType.Discrete
        assert ctx.domain_map[3].meta_type == MetaType.Continuous


class TestFeatureContextPartialDomains:
    """Test FeatureContext with partial domain specifications."""

    def test_partial_domains_dict(self):
        """Test setting domains for subset of features."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        # Only set domain for feature 1
        ctx.set_domains({1: NormalType()})

        assert ctx.domain_map[0] == MetaType.Unknown
        assert isinstance(ctx.domain_map[1], NormalType)
        assert ctx.domain_map[2] == MetaType.Unknown

    def test_incremental_domain_setting(self):
        """Test setting domains incrementally."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        # Set domains one at a time
        ctx.set_domains({0: NormalType()})
        ctx.set_domains({1: BernoulliType()})
        ctx.set_domains({2: PoissonType()})

        assert isinstance(ctx.domain_map[0], NormalType)
        assert isinstance(ctx.domain_map[1], BernoulliType)
        assert isinstance(ctx.domain_map[2], PoissonType)


class TestFeatureContextMetaTypeHandling:
    """Test handling of MetaType vs FeatureType."""

    def test_set_domains_with_meta_type(self):
        """Test set_domains with MetaType."""
        scope = Scope([0, 1])
        ctx = FeatureContext(scope)

        ctx.set_domains({0: MetaType.Continuous, 1: MetaType.Discrete})

        assert ctx.domain_map[0] == MetaType.Continuous
        assert ctx.domain_map[1] == MetaType.Discrete

    def test_mixed_meta_and_feature_types(self):
        """Test mixing MetaType and FeatureType in domains."""
        scope = Scope([0, 1, 2])
        ctx = FeatureContext(scope)

        ctx.set_domains({0: MetaType.Continuous, 1: NormalType(), 2: BernoulliType()})

        assert ctx.domain_map[0] == MetaType.Continuous
        assert isinstance(ctx.domain_map[1], NormalType)
        assert isinstance(ctx.domain_map[2], BernoulliType)
