"""Contains the Scope class for representing scopes over random variables."""

from __future__ import annotations

from collections.abc import Iterable
from functools import reduce


class Scope:
    """Scopes over random variables (RVs).

    Represents scope over random variables (RVs). Contains both RVs that are part
    of the query (i.e., RVs that are represented by the scope) as well as evidence
    of the scope (i.e., RVs that the scope is conditioned on).

    Attributes:
        query: List of non-negative integers representing query RVs.
        evidence: List of non-negative integers representing evidence variables.
    """

    def __init__(
        self,
        query: int | list[int],
        evidence: int | list[int] | None = None,
    ) -> None:
        """Initializes Scope object.

        Args:
            query: List of non-negative integers representing query RVs (may not contain duplicates).
                If a single integer is provided, it is converted to a list containing that integer.
            evidence: Optional list of non-negative integers representing evidence variables
                (may not contain duplicates or RVs that are in the query). If a single integer
                is provided, it is converted to a list containing that integer. Defaults to None,
                in which case it is initialized to an empty list.

        Raises:
            ValueError: Invalid arguments.
        """
        if query is None:
            query = []

        if evidence is None:
            evidence = []

        if isinstance(query, int):
            query = [query]

        if isinstance(evidence, int):
            evidence = [evidence]

        if len(query) == 0 and len(evidence) != 0:
            raise ValueError(
                "List of query variables for 'Scope' is empty, but list of evidence variables is not."
            )

        if any(rv < 0 for rv in query):
            raise ValueError("Query variables must all be non-negative.")

        if any(rv < 0 for rv in evidence):
            raise ValueError("Evidence variables must all be non-negative.")

        if len(query) != len(set(query)):
            raise ValueError("List of query variables for 'Scope' contains duplicates.")

        if len(evidence) != len(set(evidence)):
            raise ValueError("List of evidence variables for 'Scope' contains duplicates.")

        if not set(query).isdisjoint(evidence):
            raise ValueError("Specified query and evidence variables for 'Scope' are not disjoint.")

        self.query = query
        self.evidence = evidence

    def __repr__(self) -> str:
        """Returns a string representation of the scope of form Scope(query|evidence).

        Returns:
            String containing the string representation of scope.
        """
        if len(self.evidence) == 0:
            if len(self.query) == 1:
                return str(self.query[0])
            else:
                return str(self.query)
        else:
            if len(self.query) == 1:
                return str(self.query[0]) + "|" + str(self.evidence)
            else:
                return str(self.query) + "|" + str(self.evidence)

    def __eq__(self, other: "Scope") -> bool:
        """Equality comparison between two Scope objects.

        Two scopes are considered equal if they represent the same query and evidence RVs.

        Args:
            other: Scope object to compare to.

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

    def remove_from_query(self, rv: int) -> "Scope":
        """Removes a random variable from the query of the scope.

        Args:
            rv: Non-negative integer representing the random variable to remove.

        Returns:
            Scope object with the variable removed, or None if query becomes empty.
        """
        self.query.remove(rv)
        if len(self.query) == 0:
            return None
        else:
            return self

    def equal_query(self, other: "Scope") -> bool:
        """Checks if the query of the scope is identical to that of another.

        The order of the query RVs is not important.

        Args:
            other: Scope object to compare to.

        Returns:
            Boolean indicating whether both query scopes are identical (True) or not (False).
        """
        return set(self.query) == set(other.query)

    def equal_evidence(self, other: "Scope") -> bool:
        """Checks if the evidence of the scope is identical to that of another.

        The order of the evidence RVs is not important.

        Args:
            other: Scope object to compare to.

        Returns:
            Boolean indicating whether both evidence scopes are identical (True) or not (False).
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

        A scope is conditional if it contains evidence RVs.

        Returns:
            Boolean indicating whether the scope is conditional (True) or not (False).
        """
        return len(self.evidence) != 0

    def isdisjoint(self, other: "Scope") -> bool:
        """Checks if the scope is disjoint to another scope.

        Two scopes are considered disjoint if their queries are disjoint, i.e., they do not represent any common RVs.

        Args:
            other: Scope object to compare to.

        Returns:
            Boolean indicating whether the scopes are disjoint (True) or not (False).
        """
        return set(self.query).isdisjoint(other.query)

    def join(self, other: "Scope") -> "Scope":
            """Computes the joint scope of the scope and another scope.

            Follows probabilistic semantics, i.e.:
            - p(X) * p(Z) = p(X,Z)
            - p(X) * p(Y|Z)= p(X,Y|Z)
            - p(X) * p(Y|X)= p(X,Y)
            - p(X|Y) * p(Z|W) = p(X,Z|Y,W)
            - p(X|Y) * p(Z|Y) = p(X,Z|Y)
            - p(X|Y) * p(Y|Z) = p(X,Y|Z)

            Args:
                other: Scope object to compute the union with.

            Returns:
                Scope object representing the union of both scopes.
            """
            # Union of all query variables
            new_query = set(self.query) | set(other.query)

            # Union of evidence variables, minus those that are now in the query
            new_evidence = (set(self.evidence) | set(other.evidence)) - new_query

            return Scope(sorted(new_query), sorted(new_evidence))

    @staticmethod
    def join_all(scopes: Iterable["Scope"]) -> "Scope":
        """Computes the joint scope of the scope and a sequence of scopes.

        The union of multiple scopes results in the union of the queries and evidences, respectively.

        Args:
            scopes: Iterable of Scope objects to compute the union with.

        Returns:
            Scope object representing the union of all scopes.
        """
        return reduce(lambda a, b: a.join(b), scopes)

    @staticmethod
    def all_pairwise_disjoint(scopes: Iterable["Scope"]) -> bool:
        """Checks if a sequence of scopes are pairwise disjoint.

        Args:
            scopes: Iterable of Scope objects to check pairwise disjointness.

        Returns:
            Boolean indicating whether all scopes are pairwise disjoint (True) or not (False).
        """
        overall_scope = None

        for scope in scopes:
            if overall_scope is None:
                overall_scope = scope
                continue

            if overall_scope.isdisjoint(scope):
                overall_scope = overall_scope.join(scope)
            else:
                return False

        return True

    @staticmethod
    def all_equal(scopes: Iterable["Scope"]) -> bool:
        """Checks if a sequence of scopes are all equal.

        Args:
            scopes: Iterable of Scope objects to check for equality.

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

    def copy(self) -> "Scope":
        """Creates a copy of the scope.

        Returns:
            Scope object representing the copy of the scope.
        """
        return Scope(list(self.query), list(self.evidence))
