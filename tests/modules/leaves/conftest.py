"""Shared fixtures and matrices for leaf tests."""

from __future__ import annotations

import pytest

from tests.modules.leaves.leaf_contract_data import (
    CONDITIONAL_LEAF_CLS_VALUES,
    CONDITIONAL_LEAF_PARAMS,
    DIFF_SAMPLING_SUPPORTED_LEAF_CLS_VALUES,
    DIFFERENTIABLE_EQ_LEAF_CLS_VALUES,
    LEAF_CLS_VALUES,
    LEAF_PARAMS,
    LEAF_SAMPLE_PARAMS,
    MARGINALIZE_LEAF_PARAMS,
    NUM_REPETITION_VALUES,
    OUT_CHANNELS_VALUES,
    OUT_FEATURES_VALUES,
    TRAINABLE_LEAF_PARAMS,
)


@pytest.fixture(scope="module")
def out_channels_values():
    return OUT_CHANNELS_VALUES


@pytest.fixture(scope="module")
def out_features_values():
    return OUT_FEATURES_VALUES


@pytest.fixture(scope="module")
def num_repetition_values():
    return NUM_REPETITION_VALUES


@pytest.fixture(scope="module")
def leaf_cls_values():
    return LEAF_CLS_VALUES


@pytest.fixture(scope="module")
def differentiable_eq_leaf_cls_values():
    return DIFFERENTIABLE_EQ_LEAF_CLS_VALUES


@pytest.fixture(scope="module")
def diff_sampling_supported_leaf_cls_values():
    return DIFF_SAMPLING_SUPPORTED_LEAF_CLS_VALUES


@pytest.fixture(scope="module")
def conditional_leaf_cls_values():
    return CONDITIONAL_LEAF_CLS_VALUES


@pytest.fixture(scope="module")
def leaf_params():
    return LEAF_PARAMS


@pytest.fixture(scope="module")
def trainable_leaf_params():
    return TRAINABLE_LEAF_PARAMS


@pytest.fixture(scope="module")
def leaf_sample_params():
    return LEAF_SAMPLE_PARAMS


@pytest.fixture(scope="module")
def marginalize_leaf_params():
    return MARGINALIZE_LEAF_PARAMS


@pytest.fixture(scope="module")
def conditional_leaf_params():
    return CONDITIONAL_LEAF_PARAMS
