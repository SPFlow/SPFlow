import os
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


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """Add common modules to doctest namespace."""
    import spflow
    from spflow.meta import Scope
    from spflow.modules.leaves import Normal, Categorical
    from spflow.modules.sums import Sum
    from spflow.modules.einsum import EinsumLayer, LinsumLayer
    from spflow.modules.ops.split import SplitMode

    doctest_namespace["spflow"] = spflow
    doctest_namespace["torch"] = torch
    doctest_namespace["Scope"] = Scope
    doctest_namespace["Normal"] = Normal
    doctest_namespace["Categorical"] = Categorical
    doctest_namespace["Sum"] = Sum
    doctest_namespace["EinsumLayer"] = EinsumLayer
    doctest_namespace["LinsumLayer"] = LinsumLayer
    doctest_namespace["SplitMode"] = SplitMode

    # Common objects
    doctest_namespace["leaf"] = Normal(scope=Scope([0, 1, 2, 3]), out_channels=2)
