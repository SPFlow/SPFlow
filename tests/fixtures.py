import pytest
import torch
import os


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
