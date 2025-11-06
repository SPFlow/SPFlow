import shutil

import pytest
import torch

USE_GPU = True


def has_graphviz_dot():
    """Check if the graphviz 'dot' binary is available on the system."""
    return shutil.which("dot") is not None


@pytest.fixture(scope="session")
def device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
