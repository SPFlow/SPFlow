import pytest
import torch

USE_GPU = True

@pytest.fixture(scope="session")
def device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device