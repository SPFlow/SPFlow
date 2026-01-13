import numpy as np
import pytest
import torch

from spflow.meta import Scope
from spflow.modules.leaves import Normal


def test_scope_initialization_flexibility():
    # Test Scope with various input types
    s1 = Scope(0)
    assert s1.query == (0,)

    s2 = Scope([0, 1])
    assert s2.query == (0, 1)

    s3 = Scope(np.array([0, 1]))
    assert s3.query == (0, 1)

    s4 = Scope(torch.tensor([0, 1]))
    assert s4.query == (0, 1)

    s5 = Scope(range(3))
    assert s5.query == (0, 1, 2)


def test_normal_scope_flexibility():
    # Test Normal module with various scope input types
    n1 = Normal(scope=0)
    assert n1.scope.query == (0,)

    n2 = Normal(scope=[0])
    assert n2.scope.query == (0,)

    n3 = Normal(scope=np.arange(0, 5))
    assert n3.scope.query == (0, 1, 2, 3, 4)

    n4 = Normal(scope=torch.arange(0, 5))
    assert n4.scope.query == (0, 1, 2, 3, 4)

    n5 = Normal(scope=Scope([0, 1]))
    assert n5.scope.query == (0, 1)


def test_scope_with_evidence_flexibility():
    # Test Scope with evidence using various input types
    s1 = Scope(query=np.array([0, 1]), evidence=torch.tensor([2, 3]))
    assert s1.query == (0, 1)
    assert s1.evidence == (2, 3)

    s2 = Scope(query=range(2), evidence=range(2, 4))
    assert s2.query == (0, 1)
    assert s2.evidence == (2, 3)
