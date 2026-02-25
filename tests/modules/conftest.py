"""Shared fixtures for non-leaf module tests."""

from __future__ import annotations

import pytest

from tests.modules.module_contract_data import (
    CAT_PARAMS,
    PRODUCT_PARAMS,
    SPLIT_PARAMS,
    SUM_PARAMS,
)


@pytest.fixture(scope="session")
def sum_params():
    return SUM_PARAMS


@pytest.fixture(scope="session")
def product_params():
    return PRODUCT_PARAMS


@pytest.fixture(scope="session")
def cat_params():
    return CAT_PARAMS


@pytest.fixture(scope="session")
def split_params():
    return SPLIT_PARAMS
