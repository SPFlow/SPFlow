"""Contains the ``Scope`` class for representing scopes over random variables.

Typical usage example:

    scope = Scope(query_rvs, evidence_rvs)
"""
from typing import Iterable, List, Optional, Type, Union

from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType


class Scope:
    """Scopes over random variables (RVs).

    Represents scope over random variables (RVs).
    Contains both RVs that are part of the query (i.e., RVs that are represented by the scope) as well as evidence of the scope (i.e., RVs that the scope is conditioned on).

    Attributes:
        query:
            List of non-negative integers representing query RVs.
        evidence:
            List of non-negative integers representing evidence variables.
    """

    def __init__(
        self,
        query: Optional[List[int]] = None,
        evidence: Optional[List[int]] = None,
    ) -> None:
        """Initializes ``Scope`` object.

        Args:
            query:
                List of non-negative integers representing query RVs (may not contain duplicates).
                Defaults to None, in which case it is initialized to an empty list.
            evidence:
                Optional list of non-negative integers representing evidence variables (may not contain duplicates or RVs that are in the query).
                Defaults to None, in which case it is initialized to an empty list.

        Raises:
            ValueError: Invalid arguments.
        """
        if query is None:
            query = []

        if evidence is None:
            evidence = []

        if len(query) == 0 and len(evidence) != 0:
            raise ValueError(
                "List of query variables for 'Scope' is empty, but list of evidence variables is not."
            )

        if any(rv < 0 for rv in query):
            raise ValueError("Query variables must all be non-negative.")

        if any(rv < 0 for rv in evidence):
            raise ValueError("Evidence variables must all be non-negative.")

        if len(query) != len(set(query)):
            raise ValueError(
                "List of query variables for 'Scope' contains duplicates."
            )

        if len(evidence) != len(set(evidence)):
            raise ValueError(
                "List of evidence variables for 'Scope' contains duplicates."
            )

        if not set(query).isdisjoint(evidence):
            raise ValueError(
                "Specified query and evidence variables for 'Scope' are not disjoint."
            )

        self.query = query
        self.evidence = evidence

    def __repr__(self) -> str:
        """Returns a string representation of the scope of form ``Scope(query|evidence)``.

        Returns:
            String containg the string representation of scope.
        """
        return "Scope({}|{})".format(
            self.query if self.query else "{}",
            self.evidence if self.evidence else "{}",
        )  # pragma: no cover

    def __eq__(self, other: "Scope") -> bool:
        """Equality comparison between two ``Scope`` objects.

        Two scopes are considered equal if they represent the same query and evidence RVs.

        Args:
            other:
                ``Scope`` object to compare to.

        Returns:
            Boolean indicating whether both scopes are considered equal (True) or not (False).
        """
        return self.equal_query(other) and self.equal_evidence(other)

    def __len__(self) -> int:
        """Returns the number of query variables in the scope.

        Returns:
            Integer representing the number of query variables.
        """
        return len(self.query)

    def equal_query(self, other: "Scope") -> bool:
        """Checks if the query of the scope is identical to that of another.

        The order of the query RVs is not important.

        Args:
            other:
                ``Scope` object to compare to.

        Returns:
            Boolean indicating whether both query scopes are idential (True) or not (False).
        """
        return set(self.query) == set(other.query)

    def equal_evidence(self, other: "Scope") -> bool:
        """Checks if the evidence of the scope is identical to that of another.

        The order of the evidence RVs is not important.

        Args:
            other:
                ``Scope`` object to compare to.

        Returns:
            Boolean indicating whether both evidence scopes are idential (True) or not (False).
        """
        return set(self.evidence) == set(other.evidence)

    def isempty(self) -> bool:
        """Checks if the scope is empty.

        A scope is considered empty if its query is empty, i.e., the scope does not represent any RVs.

        Returns:
            Boolean indicating whether the scope is empty (True) or not (False).
        """
        return not bool(self.query)

    def is_conditional(self) -> bool:
        """Checks if the scope is conditional.

        A scope is conditional, if it contains evidence RVs

        Returns:
            Boolean indicating whether the scope is conditional (True) or not (False).
        """
        return len(self.evidence) != 0

    def isdisjoint(self, other: "Scope") -> bool:
        """Checks if the scope is disjoint to another scope.

        Two scopes are considered disjoint if their queries are disjoint, i.e., they do not represent any common RVs.

        Returns:
            Boolean indicating whether the scopes are disjoint (True) or not (False).
        """
        return set(self.query).isdisjoint(other.query)

    def join(self, other: "Scope") -> "Scope":
        """Computes the joint scope of the scope and another scope.

        The union of two scopes results in the union of the queries and evidences, respectively.

        Args:
            other:
                ``Scope`` object to compute the union with.

        Returns:
            ``Scope`` object representing the union of both scopes.
        """
        # compute union of query RVs
        joint_query = list(set(self.query).union(other.query))
        # compute union of evidence RVs
        joint_evidence = list(set(self.evidence).union(other.evidence))

        return Scope(joint_query, joint_evidence)

    @staticmethod
    def all_pairwise_disjoint(scopes: Iterable["Scope"]) -> bool:
        """Checks if a sequence of scopes are pairwise disjoint.

        Args:
            scopes:
                Iterable of ``Scope`` objects to check pairwise disjointness.

        Returns:
            Boolean indicating whether all scopes are pairwise disjoint (True) or not (False).
        """
        overall_scope = Scope()

        for scope in scopes:
            if overall_scope.isdisjoint(scope):
                overall_scope = overall_scope.join(scope)
            else:
                return False

        return True

    @staticmethod
    def all_equal(scopes: Iterable["Scope"]) -> bool:
        """Checks if a sequence of scopes are all equal.

        Args:
            scopes:
                Iterable of ``Scope`` objects to check for equality.

        Returns:
            Boolean indicating whether all scopes are equal (True) or not (False).
        """
        overall_scope = None

        for scope in scopes:
            if overall_scope is None:
                overall_scope = scope
                continue
            if not overall_scope == scope:
                return False

        return True
