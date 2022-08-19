"""
Created on August 03, 2022

@authors: Philipp Deibert
"""
from typing import List, Optional, Iterable


class Scope():
    """Class representing Variable scopes.
    
    Args:
        query: list of integers representing query variables (may not be empty or contain duplicates).
        evidence: (optional) list of integers representing evidence variables (may be empty; may not contain duplicates).
    """
    def __init__(self, query: Optional[List[int]]=None, evidence: Optional[List[int]]=None) -> None:
        """TODO."""
        
        if query is None:
            query = set()

        if evidence is None:
            evidence = set()

        if len(query) != len(set(query)):
            raise ValueError("List of query variables for 'Scope' contains duplicates.")

        if len(evidence) != len(set(evidence)):
            raise ValueError("List of evidence variables for 'Scope' contains duplicates.")

        if not set(query).isdisjoint(evidence):
            raise ValueError("Specified query and evidence variables for 'Scope' are not disjoint.")

        self.query = query
        self.evidence = evidence

    def __repr__(self) -> str:
        """TODO"""
        return "Scope({}|{})".format(self.query if self.query else "{}", self.evidence if self.evidence else "{}")

    def __eq__(self, other) -> bool:
        """TODO"""
        return self.equal_query(other) and self.equal_evidence(other) 

    def __len__(self) -> int:
        """Returns the number of query variables"""
        return len(self.query)
    
    def equal_query(self, other) -> bool:
        """TODO"""
        return (set(self.query) == set(other.query))
    
    def equal_evidence(self, other) -> bool:
        """TODO"""
        return (set(self.evidence) == set(other.evidence))

    def isempty(self) -> bool:
        return not bool(self.query)

    def isdisjoint(self, other) -> bool:
        """TODO"""
        return set(self.query).isdisjoint(other.query)

    def union(self, other) -> "Scope":
        """TODO"""
        return Scope(set(self.query).union(other.query), set(self.evidence).union(other.evidence))

    def all_pairwise_disjoint(scopes: Iterable["Scope"]) -> bool:
    
        overall_scope = Scope()

        for scope in scopes:
            if(overall_scope.isdisjoint(scope)):
                overall_scope = overall_scope.union(scope)
            else:
                return False
    
        return True
    
    def all_equal(scopes: Iterable["Scope"]) -> bool:

        overall_scope = None

        for scope in scopes:
            if overall_scope is None:
                overall_scope == None
            if not overall_scope == scope:
                return False
        
        return True
