"""
Created on May 05, 2021

@authors: Kevin Huy Nguyen, Bennet Wittelsbach, Philipp Deibert

This file provides the structure and construction algorithm for abstract RegionGraphs, which are
used to build RAT-SPNs.
"""
import random
import numpy as np
from typing import Optional, Set, List
from spflow.meta.scope.scope import Scope


class Region:
    """TODO"""
    def __init__(self, scope: Scope, partitions: List["Partition"]=[]):

        if len(scope.query) == 0:
            raise ValueError("Query scope for 'Region' is empty.")

        self.scope = scope
        self.partitions = partitions


class Partition:
    """TODO"""
    def __init__(self, scope: Scope, regions: List["Region"]):

        if len(scope.query) == 0:
            raise ValueError("Query scope for 'Partition' is empty.")

        if len(regions) == 0:
            raise ValueError("List of regions for 'Partition' is empty.")

        self.scope = scope
        self.regions = regions


class RegionGraph():
    """TODO"""
    def __init__(self, root_region: Optional[Region]=None):

        if not isinstance(root_region, Region) and root_region is not None:
            raise ValueError(f"'RegionGraph' expects root region of type 'Region' or 'None', but got object of type {type(root_region)}.")
        
        self.root_region = root_region
        self.scope = root_region.scope if root_region is not None else Scope([])


def random_region_graph(
    scope: Scope, depth: int, replicas: int, n_splits: int = 2
) -> RegionGraph:
    """Creates a region graph composed of Regions and Partitions.

    This algorithm is an implementation of "Algorithm 1" of the original paper.

    Args:
        scope:
            TODO
        depth:
            (D in the paper)
            An integer that controls the depth of the graph structure of the RegionGraph. One level
            of depth equals to a pair of (Partitions, Regions). The root has depth 0.
        replicas:
            (R in the paper)
            An integer for the number of replicas. Replicas are distinct Partitions of the whole
            set of random variables X, which are children of the root_region of the RegionGraph.
        n_splits:
            The number of splits per Region (defaults to 2).

    Returns:
        A RegionGraph with a binary tree structure, consisting of alternating Regions and Partitions.

    Raises:
        ValueError: If any argument is invalid.
    """
    if len(scope.query) < n_splits:
        raise ValueError("Need at least 'n_splits' query RVs to build region graph.")
    if depth < 0:
        raise ValueError("Depth must not be negative.")
    if replicas < 1:
        raise ValueError("Number of replicas must be at least 1.")
    if n_splits < 2:
        raise ValueError("Number of splits must be at least 2.")
    
    if depth >= 1:
        partitions = [split(scope, depth, n_splits) for r in range(replicas)]
    else:
        partitions = []

    root_region = Region(scope=scope, partitions=partitions)

    return RegionGraph(root_region)


def split(
    scope: Scope,
    depth: int,
    n_splits: int = 2
) -> Partition:
    """Splits a scope into (currently balanced) Partitions.

    TODO:
    Recursively builds up a binary tree structure of the region graph. First, it splits the
    random variables of the parent_region, Y, into a Partition consisting of two balanced,
    distinct subsets of Y, and adds it as a child of the parent_region. Then, split() will
    be called onto each of the two subsets of Y until the maximum depth of the RegionGraph
    is reached, OR each Region consists of only 1 random variable,

    Args:
        partition_scope:
            TODO
        depth:
            The maximum depth of the region graph until which split() will be recursively called.
        n_splits:
            The number of splits per Region.
    """
    if n_splits < 2:
        raise ValueError(f"Number of splits must be at least 2, but is {n_splits}.")
    
    if depth < 1:
        raise ValueError(f"Depth for splitting scope is expected to be at least 1, but is {depth}.")

    shuffled_rvs = scope.query.copy()
    random.shuffle(shuffled_rvs)

    split_rvs = np.array_split(shuffled_rvs, n_splits)

    if any(region_rvs.size == 0 for region_rvs in split_rvs):
        raise ValueError(
            "Number of query scope variables cannot be split into 'n_splits' non-empty splits (make sure 'split' is called appropriately)."
        )

    regions = []

    for region_rvs in split_rvs:
        region_scope = Scope(region_rvs.tolist(), scope.evidence)

        partitions = []
        new_depth = depth - 1

        # create partition if region scope can be divided into 'n_splits' non-empty splits and depth limit not reached yet, otherwise this is a terminal leaf region
        if new_depth > 0 and len(region_rvs) >= n_splits:
            partitions = [split(region_scope, new_depth, n_splits)]        

        regions.append(Region(region_scope, partitions=partitions))

    return Partition(scope, regions=regions)