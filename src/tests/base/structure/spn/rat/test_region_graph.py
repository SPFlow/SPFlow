from typing import Tuple
from spflow.meta.data.scope import Scope
from spflow.base.structure.spn.rat.region_graph import (
    Region,
    Partition,
    RegionGraph,
    random_region_graph,
)
import unittest


def get_region_graph_properties(
    region_graph: RegionGraph,
) -> Tuple[int, int, int, int]:

    correct_scopes = True
    max_depth = 0
    n_internal_regions = 0
    n_partitions = 0
    n_leaf_regions = 0

    regions = [(region_graph.root_region, 0)]

    while regions:
        region, depth = regions.pop()

        if region.partitions:
            n_internal_regions += 1

            for partition in region.partitions:
                n_partitions += 1

                split_scope = Scope([])

                for r in partition.regions:
                    regions.append((r, depth + 1))

                    # make sure child regions of partition have pair-wise dijoint scopes
                    correct_scopes &= split_scope.isdisjoint(r.scope)
                    # update joint scope
                    split_scope = split_scope.join(r.scope)

                # make sure joint scope matches partition scope
                correct_scopes &= split_scope == partition.scope

                # make sure partition scope matches parent region's scope
                correct_scopes &= partition.scope == region.scope
        else:
            n_leaf_regions += 1
            max_depth = max(max_depth, depth)

    return (
        correct_scopes,
        max_depth,
        n_internal_regions,
        n_partitions,
        n_leaf_regions,
    )


class TestRegionGraph(unittest.TestCase):
    def test_leaf_region_initialization(self):

        Region(Scope([0, 1]))
        # empty scope
        self.assertRaises(ValueError, Region, Scope([]))

    def test_partition_initialization(self):

        leaf_regions = [Region(Scope([0, 1])), Region(Scope([2]))]
        Partition(Scope([0, 1]), leaf_regions)

        # no regions
        self.assertRaises(TypeError, Partition, Scope([0, 1]))
        # empty scope
        self.assertRaises(ValueError, Partition, Scope([]), leaf_regions)

    def test_region_initialization(self):

        leaf_regions = [Region(Scope([0, 1])), Region(Scope([2]))]
        partition = Partition(Scope([0, 1]), leaf_regions)

        Region(Scope([0, 1, 2]), [partition])

    def test_random_region_graph_none(self):

        depth = 1
        replicas = 1
        n_splits = 2

        self.assertRaises(
            AttributeError,
            random_region_graph,
            scope=None,
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

    def test_random_region_graph_not_enough_randomvars(self):

        for num_random_variables in range(3):
            random_variables = list(range(num_random_variables))
            depth = 1
            replicas = 1
            n_splits = num_random_variables + 1

            self.assertRaises(
                ValueError,
                random_region_graph,
                scope=Scope(random_variables),
                depth=depth,
                replicas=replicas,
                n_splits=n_splits,
            )

    def test_random_region_graph_negative_depth(self):
        random_variables = list(range(2))
        depth = -1
        replicas = 1
        n_splits = 2

        self.assertRaises(
            ValueError,
            random_region_graph,
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

    def test_random_region_graph_no_replicas(self):
        random_variables = list(range(2))
        depth = 1
        replicas = 0
        n_splits = 2

        self.assertRaises(
            ValueError,
            random_region_graph,
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

    def test_random_region_graph_invalid_num_split(self):
        random_variables = list(range(2))
        depth = 1
        replicas = 0
        n_splits = 1

        self.assertRaises(
            ValueError,
            random_region_graph,
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

    def test_random_region_graph_structure_1(self):
        random_variables = list(range(7))
        depth = 2
        replicas = 1
        n_splits = 2

        region_graph = random_region_graph(
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

        # get region graph properties
        (
            correct_scopes,
            max_depth,
            n_regions,
            n_partitions,
            n_leaf_regions,
        ) = get_region_graph_properties(region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(max_depth, 2)
        # assert correct number of intern regions, partitions, and leaf regions
        self.assertEqual(n_regions, 3)
        self.assertEqual(n_partitions, 3)
        self.assertEqual(n_leaf_regions, 4)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertTrue(correct_scopes)

    def test_region_graph_structure_2(self):
        random_variables = list(range(7))  # 7 randomvars
        depth = 3
        replicas = 1
        n_splits = 2

        region_graph = random_region_graph(
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

        # get region graph properties
        (
            correct_scopes,
            max_depth,
            n_regions,
            n_partitions,
            n_leaf_regions,
        ) = get_region_graph_properties(region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(max_depth, 3)
        # assert correct number of intern regions, partitions, and leaf regions
        self.assertEqual(n_regions, 6)
        self.assertEqual(n_partitions, 6)
        self.assertEqual(n_leaf_regions, 7)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertTrue(correct_scopes)

    def test_region_graph_structure_3(self):
        random_variables = list(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 2
        n_splits = 2

        region_graph = random_region_graph(
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

        # get region graph properties
        (
            correct_scopes,
            max_depth,
            n_regions,
            n_partitions,
            n_leaf_regions,
        ) = get_region_graph_properties(region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(max_depth, 3)
        # assert correct number of intern regions, partitions, and leaf regions
        self.assertEqual(n_regions, 11)
        self.assertEqual(n_partitions, 12)
        self.assertEqual(n_leaf_regions, 14)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertTrue(correct_scopes)

    def test_region_graph_structure_4(self):
        random_variables = list(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 2
        n_splits = 3

        region_graph = random_region_graph(
            scope=Scope(random_variables),
            depth=depth,
            replicas=replicas,
            n_splits=n_splits,
        )

        # get region graph properties
        (
            correct_scopes,
            max_depth,
            n_regions,
            n_partitions,
            n_leaf_regions,
        ) = get_region_graph_properties(region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(max_depth, 2)
        # assert correct number of intern regions, partitions, and leaf regions
        self.assertEqual(n_regions, 3)
        self.assertEqual(n_partitions, 4)
        self.assertEqual(n_leaf_regions, 10)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertTrue(correct_scopes)


if __name__ == "__main__":
    unittest.main()
