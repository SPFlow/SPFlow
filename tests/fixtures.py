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
