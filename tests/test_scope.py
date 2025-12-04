import pytest

from spflow.meta import Scope


def test_equal_query():
    # Test if queries are equal regardless of order
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])  # Same query as scope1, different order
    scope3 = Scope([1, 2, 4], [3, 5])  # Different query

    assert scope1.equal_query(scope2)  # Should be equal
    assert not scope1.equal_query(scope3)  # Should not be equal


def test_equal_evidence():
    # Test if evidence is equal regardless of order
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])  # Same evidence as scope1, different order
    scope3 = Scope([1, 2, 3], [4, 6])  # Different evidence

    assert scope1.equal_evidence(scope2)  # Should be equal
    assert not scope1.equal_evidence(scope3)  # Should not be equal


def test_is_conditional():
    # Test for conditional and non-conditional scopes
    conditional_scope = Scope([1, 2], [3, 4])  # Has evidence
    non_conditional_scope = Scope([1, 2])  # No evidence

    assert conditional_scope.is_conditional()  # Should be conditional
    assert not non_conditional_scope.is_conditional()  # Should not be conditional


def test_isdisjoint():
    # Test for disjoint and non-disjoint scopes
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([4, 5, 6], [7, 8])  # Disjoint with scope1
    scope3 = Scope([3, 4, 5], [6, 7])  # Not disjoint with scope1

    assert scope1.isdisjoint(scope2)  # Should be disjoint
    assert not scope1.isdisjoint(scope3)  # Should not be disjoint


def test_join():
    # Test joining two scopes
    scope1 = Scope([1, 2], [6, 7])
    scope2 = Scope([2, 3], [7, 8])
    joined_scope = scope1.join(scope2)

    # Check if the joined scope has the correct query and evidence
    assert set(joined_scope.query) == {1, 2, 3}
    assert set(joined_scope.evidence) == {6, 7, 8}


def test_all_pairwise_disjoint():
    # Test if a set of scopes are all pairwise disjoint
    scope1 = Scope([1, 2])
    scope2 = Scope([3, 4])
    scope3 = Scope([1, 6])  # Not disjoint with scope1

    assert Scope.all_pairwise_disjoint([scope1, scope2])  # Should be all disjoint
    assert not Scope.all_pairwise_disjoint([scope1, scope2, scope3])  # Should not be all disjoint


def test_all_equal():
    # Test if a set of scopes are all equal
    scope1 = Scope([1, 2, 3], [4, 5])
    scope2 = Scope([3, 2, 1], [5, 4])  # Same as scope1, different order
    scope3 = Scope([1, 2, 3], [4, 5])  # Same as scope1
    scope4 = Scope([1, 2, 4], [3, 5])  # Different from others

    assert Scope.all_equal([scope1, scope2, scope3])  # Should all be equal
    assert not Scope.all_equal([scope1, scope2, scope3, scope4])  # Should not all be equal


def test_copy():
    # Test copying a scope
    original_scope = Scope([1, 2, 3], [4, 5])
    copied_scope = original_scope.copy()

    # Check if the copy is equal but not the same object
    assert original_scope == copied_scope
    assert original_scope is not copied_scope

    # Verify that modification is prevented (frozen dataclass)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        copied_scope.query = (1, 2, 3, 4)


def test_scope_init_valid():
    # Test valid initializations
    scope1 = Scope([1, 2, 3], [4, 5])
    assert scope1.query == (1, 2, 3)
    assert scope1.evidence == (4, 5)

    scope2 = Scope([0, 1])  # Only query, no evidence
    assert scope2.query == (0, 1)
    assert scope2.evidence == ()


def test_scope_init_invalid():
    # Test invalid initializations

    # Empty query with non-empty evidence
    with pytest.raises(
        ValueError,
        match="List of query variables for 'Scope' is empty, but list of evidence variables is not.",
    ):
        Scope([], [1, 2])

    # Negative values in query
    with pytest.raises(ValueError, match="Query variables must all be non-negative."):
        Scope([-1, 0, 1])

    # Negative values in evidence
    with pytest.raises(ValueError, match="Evidence variables must all be non-negative."):
        Scope([0, 1], [-1, 2])

    # Duplicate values in query
    with pytest.raises(ValueError, match="List of query variables for 'Scope' contains duplicates."):
        Scope([1, 2, 2, 3])

    # Duplicate values in evidence
    with pytest.raises(ValueError, match="List of evidence variables for 'Scope' contains duplicates."):
        Scope([1, 2, 3], [4, 4, 5])

    # Overlapping query and evidence
    with pytest.raises(
        ValueError, match="Specified query and evidence variables for 'Scope' are not disjoint."
    ):
        Scope([1, 2, 3], [3, 4, 5])
