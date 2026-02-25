import pytest

from spflow.meta import Scope


def test_equal_query():
    # Query equality is set-based, so permutation must not change semantics.
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])
    scope3 = Scope([1, 2, 4], [3, 5])

    assert scope1.equal_query(scope2)
    assert not scope1.equal_query(scope3)


def test_equal_evidence():
    # Evidence comparison should also be order-insensitive.
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])
    scope3 = Scope([1, 2, 3], [4, 6])

    assert scope1.equal_evidence(scope2)
    assert not scope1.equal_evidence(scope3)


def test_is_conditional():
    # Conditionality is derived only from whether evidence is present.
    conditional_scope = Scope([1, 2], [3, 4])
    non_conditional_scope = Scope([1, 2])

    assert conditional_scope.is_conditional()
    assert not non_conditional_scope.is_conditional()


def test_isdisjoint():
    # Disjointness is checked across query+evidence, not query alone.
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([4, 5, 6], [7, 8])
    scope3 = Scope([3, 4, 5], [6, 7])

    assert scope1.isdisjoint(scope2)
    assert not scope1.isdisjoint(scope3)


def test_join():
    # Join should form unions while preserving uniqueness invariants.
    scope1 = Scope([1, 2], [6, 7])
    scope2 = Scope([2, 3], [7, 8])
    joined_scope = scope1.join(scope2)

    assert set(joined_scope.query) == {1, 2, 3}
    assert set(joined_scope.evidence) == {6, 7, 8}


def test_all_pairwise_disjoint():
    # Pairwise checks must fail fast when any overlap appears.
    scope1 = Scope([1, 2])
    scope2 = Scope([3, 4])
    scope3 = Scope([1, 6])

    assert Scope.all_pairwise_disjoint([scope1, scope2])
    assert not Scope.all_pairwise_disjoint([scope1, scope2, scope3])


def test_all_equal():
    # Equality helper should ignore ordering but reject semantic mismatches.
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])
    scope3 = Scope([1, 2, 3], [4, 5])
    scope4 = Scope([1, 2, 4], [3, 5])

    assert Scope.all_equal([scope1, scope2, scope3])
    assert not Scope.all_equal([scope1, scope2, scope3, scope4])


def test_copy():
    # Copy should preserve value semantics without aliasing instances.
    original_scope = Scope([1, 2, 3], [4, 5])
    copied_scope = original_scope.copy()

    assert original_scope == copied_scope
    assert original_scope is not copied_scope

    # Immutability is part of Scope's contract.
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        copied_scope.query = (1, 2, 3, 4)


def test_scope_init_valid():
    scope1 = Scope([1, 2, 3], [4, 5])
    assert scope1.query == (1, 2, 3)
    assert scope1.evidence == (4, 5)

    # Empty evidence should normalize to an empty tuple.
    scope2 = Scope([0, 1])
    assert scope2.query == (0, 1)
    assert scope2.evidence == ()


def test_scope_init_invalid():
    # These guards protect canonical Scope invariants expected by graph logic.
    with pytest.raises(
        ValueError,
    ):
        Scope([], [1, 2])

    with pytest.raises(ValueError):
        Scope([-1, 0, 1])

    with pytest.raises(ValueError):
        Scope([0, 1], [-1, 2])

    with pytest.raises(ValueError):
        Scope([1, 2, 2, 3])

    with pytest.raises(ValueError):
        Scope([1, 2, 3], [4, 4, 5])

    with pytest.raises(ValueError):
        Scope([1, 2, 3], [3, 4, 5])
