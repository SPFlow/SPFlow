from __future__ import annotations

import numpy as np
import pytest

from spflow.exceptions import InvalidParameterError
from spflow.meta import Scope
from spflow.modules.leaves import Normal
from spflow.modules.ops import SplitMode
from spflow.zoo.rat import RatSPN
from tests.utils.leaves import make_leaf
from tests.zoo.rat.rat_test_utils import DummyLeaf

pytestmark = pytest.mark.contract


def test_constructor_rejects_invalid_hyperparameters():
    valid_leaf = make_leaf(cls=Normal, out_channels=2, out_features=4, num_repetitions=1)

    with pytest.raises(InvalidParameterError):
        RatSPN(leaf_modules=[valid_leaf], n_root_nodes=0, n_region_nodes=1, num_repetitions=1, depth=1)
    with pytest.raises(InvalidParameterError):
        RatSPN(leaf_modules=[valid_leaf], n_root_nodes=1, n_region_nodes=0, num_repetitions=1, depth=1)
    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[DummyLeaf(channels=0)], n_root_nodes=1, n_region_nodes=1, num_repetitions=1, depth=1
        )
    with pytest.raises(InvalidParameterError):
        RatSPN(
            leaf_modules=[valid_leaf],
            n_root_nodes=1,
            n_region_nodes=1,
            num_repetitions=1,
            depth=1,
            split_mode=SplitMode.consecutive(),
            num_splits=1,
        )


def test_rat_spn_feature_to_scope():
    num_features = 64
    leaf_layer = make_leaf(cls=Normal, out_channels=3, out_features=num_features, num_repetitions=1)
    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=2,
        n_region_nodes=4,
        num_repetitions=1,
        depth=1,
        outer_product=False,
        split_mode=SplitMode.consecutive(),
    )

    feature_scopes = model.feature_to_scope
    # feature_to_scope is a delegated view used by tooling; keep it identical to root metadata.
    assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
    assert feature_scopes.shape == (1, 1)
    assert feature_scopes[0, 0] == model.scope
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_single_root_node():
    leaf_layer = make_leaf(cls=Normal, out_channels=2, out_features=4, num_repetitions=1)
    model = RatSPN(
        leaf_modules=[leaf_layer],
        n_root_nodes=1,
        n_region_nodes=3,
        num_repetitions=1,
        depth=1,
        outer_product=False,
        split_mode=SplitMode.interleaved(),
    )

    feature_scopes = model.feature_to_scope
    assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
    assert len(feature_scopes[0, 0].query) == 4
    assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_multiple_repetitions():
    num_features = 64
    for num_reps in [1, 2, 3]:
        leaf_layer = make_leaf(
            cls=Normal, out_channels=4, out_features=num_features, num_repetitions=num_reps
        )
        model = RatSPN(
            leaf_modules=[leaf_layer],
            n_root_nodes=2,
            n_region_nodes=3,
            num_repetitions=num_reps,
            depth=1,
            outer_product=False,
            split_mode=SplitMode.consecutive(),
        )
        feature_scopes = model.feature_to_scope
        assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
        assert len(feature_scopes[0, 0].query) == num_features
        assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())


def test_rat_spn_feature_to_scope_split_variants():
    num_features = 64
    for split_mode_val in [None, SplitMode.consecutive(), SplitMode.interleaved()]:
        # Split strategy should only change internal regioning, not exposed scope coverage.
        leaf_layer = make_leaf(cls=Normal, out_channels=3, out_features=num_features, num_repetitions=1)
        model = RatSPN(
            leaf_modules=[leaf_layer],
            n_root_nodes=2,
            n_region_nodes=4,
            num_repetitions=1,
            depth=1,
            outer_product=False,
            split_mode=split_mode_val,
        )
        feature_scopes = model.feature_to_scope
        assert np.array_equal(feature_scopes, model.root_node.feature_to_scope)
        assert len(feature_scopes[0, 0].query) == num_features
        assert all(isinstance(scope_obj, Scope) for scope_obj in feature_scopes.flatten())
