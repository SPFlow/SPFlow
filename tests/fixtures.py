import os

import pytest
import torch


@pytest.fixture(
    scope="function",
    autouse=True,
)
def auto_set_test_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


@pytest.fixture(
    scope="function",
    autouse=True,
)
def auto_set_test_device():
    device = os.getenv("SPFLOW_TEST_DEVICE", "cpu")
    assert device == "cpu" or "cuda" in device, "SPFLOW_TEST_DEVICE must be 'cpu' or 'cuda' or 'cuda:<id>'"
    torch.set_default_device(device)
    print(f"Using device: {device}")
    yield


@pytest.fixture(
    scope="function",
    autouse=True,
)
def auto_reset_dtype():
    """Reset PyTorch default dtype after each test to prevent test isolation issues.

    This fixture prevents tests that modify the global default dtype from affecting
    subsequent tests. This is critical for the parameter descriptor tests which expect
    specific dtype values.
    """
    original_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(original_dtype)
