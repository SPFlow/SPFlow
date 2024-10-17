# """Contains classes and functions to create Region Graphs.
# """
# import random
# from typing import List, Optional

# from spflow.utils import Tensor
# from spflow import tensor as T

# from spflow.meta.data.scope import Scope


# class Region:
#     r"""Represents an abstract region as part of a region graph.

#     For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

#     Attributes:
#        scope:
#             Scope represented by the region.
#         partitions:
#             List of ``Partition`` objects belonging to the region.
#     """

#     def __init__(self, scope: Scope, partitions: list["Partition"] = None) -> None:
#         r"""Initializes 'Region' object.

#         Args:
#             scope:
#                 Scope represented by the region.
#             partitions:
#                 Optional list of ``Partition`` objects belonging to the region.
#                 Defaults to ``None``, in which case it is initialized to an empty list.

#         Raises:
#             ValueError: Invalid arguments.
#         """
#         if len(scope.query) == 0:
#             raise ValueError("Query scope for 'Region' is empty.")

#         if partitions is None:
#             partitions = []

#         self.scope = scope
#         self.partitions = partitions


# class Partition:
#     r"""Represents an abstract partition as part of a region graph.

#     For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

#     Attributes:
#         scope:
#             Scope represented by the partition.
#         regions:
#             List of ``Region`` objects belonging to the partition.
#     """

#     def __init__(self, scope: Scope, regions: list["Region"]) -> None:
#         r"""Initializes 'Partition' object.

#         Args:
#             scope:
#                 Scope represented by the partition.
#             regions:
#                 List of ``Region`` objects belonging to the partition.

#         Raises:
#             ValueError: Invalid arguments.
#         """
#         if len(scope.query) == 0:
#             raise ValueError("Query scope for 'Partition' is empty.")

#         if len(regions) == 0:
#             raise ValueError("List of regions for 'Partition' is empty.")

#         self.scope = scope
#         self.regions = regions


# class RegionGraph:
#     r"""Abstract region graph consisting of abstract regions and partitions.

#     For details see (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

#     Attributes:
#         scope:
#             Scope represented by the region graph.
#         root_region:
#             ``Region`` object representing the root (i.e., top-most) region of the graph.
#     """

#     def __init__(self, root_region: Optional[Region] = None) -> None:
#         r"""Initializes 'RegionGraph' object.

#         Args:
#             root_region:
#                 ``Region`` object representing the root (i.e., top-most) region of the graph.

#         Raises:
#             ValueError: Invalid arguments.
#         """
#         if not isinstance(root_region, Region) and root_region is not None:
#             raise ValueError(
#                 f"'RegionGraph' expects root region of type 'Region' or 'None', but got object of type {type(root_region)}."
#             )

#         self.root_region = root_region
#         self.scope = root_region.scope if root_region is not None else Scope([])


# def random_region_graph(scope: Scope, depth: int, replicas: int, n_splits: int = 2) -> RegionGraph:
#     r"""Creates a random instance of a region graph.

#     For details see "Algorithm 1" in (Peharz et al., 2020): "Random Sum-Product Networks: A Simple and Effective Approach to Probabilistic Deep Learning".

#     Args:
#         scope:
#             Scope to be represented by the region graph.
#         depth:
#             Integer specifying the depth of the region graph (D in the original paper).
#             The root region has depth 0. Any additional level is represented by a pair of partition and subsequent region(s).
#         replicas:
#             Integer specifying the number of replicas to be created (R in the original paper).
#             Replicas are distinct partitions over the same scope that are combined by the root region.
#         n_splits:
#             Integer specifying the number of scope partitions at each partition.
#             Defaults to 2.

#     Returns:
#         A ``RegionGraph`` instance with tree structure, consisting of a root region and alternating subsequent partitions and regions.

#     Raises:
#         ValueError: If any argument is invalid.
#     """
#     if len(scope.query) < n_splits:
#         raise ValueError("Need at least 'n_splits' query RVs to build region graph.")
#     if depth < 0:
#         raise ValueError("Depth must not be negative.")
#     if replicas < 1:
#         raise ValueError("Number of replicas must be at least 1.")
#     if n_splits < 2:
#         raise ValueError("Number of splits must be at least 2.")

#     if depth >= 1:
#         partitions = [split(scope, depth, n_splits) for r in range(replicas)]
#     else:
#         partitions = []

#     root_region = Region(scope=scope, partitions=partitions)

#     return RegionGraph(root_region)


# def split(scope: Scope, depth: int, n_splits: int = 2) -> Partition:
#     r"""Creates a ``Partion`` instance for a specified scopes.

#     Splits a specified scope into a given number of approximately equal sized parts (i.e., regions)
#     and recusively creates subsequent partitions with decreasing value for ``depth`` until
#     the scope cannot be partitioned into ``n_splits`` parts anymore or ``depth`` reaches zero.

#     Args:
#         scope:
#             Scope to be represented by the region graph.
#         depth:
#             Integer specifying the depth of the partition (D in the original paper).
#         n_splits:
#             Integer specifying the number of approximately equal sized scope partitions.
#             Defaults to 2.

#     Returns:
#         A ``Partition`` instance with recursively created sub-regions and partitions.

#     Raises:
#         ValueError: If any argument is invalid.
#     """
#     if n_splits < 2:
#         raise ValueError(f"Number of splits 'n_splits' must be at least 2, but is {n_splits}.")

#     if depth < 1:
#         raise ValueError(f"Depth for splitting scope 'depth' is expected to be at least 1, but is {depth}.")

#     shuffled_rvs = scope.query.copy()
#     random.shuffle(shuffled_rvs)

#     split_rvs = torch.array_split(shuffled_rvs, n_splits)

#     if any(region_rvs.size == 0 for region_rvs in split_rvs):
#         raise ValueError(
#             "Number of query scope variables cannot be split into 'n_splits' non-empty splits (make sure 'split' is called appropriately)."
#         )

#     regions = []

#     for region_rvs in split_rvs:
#         region_scope = Scope(region_rvs.tolist(), scope.evidence)

#         partitions = []
#         new_depth = depth - 1

#         # create partition if region scope can be divided into 'n_splits' non-empty splits and depth limit not reached yet, otherwise this is a terminal leaf region
#         if new_depth > 0 and len(region_rvs) >= n_splits:
#             partitions = [split(region_scope, new_depth, n_splits)]

#         regions.append(Region(region_scope, partitions=partitions))

#     return Partition(scope, regions=regions)
