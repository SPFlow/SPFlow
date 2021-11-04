from spflow.base.structure.rat.region_graph import random_region_graph, _get_regions_by_depth
import unittest


class TestRegionGraph(unittest.TestCase):
    def test_region_graph_empty_randomvars(self):
        random_variables = None
        depth = 1
        replicas = 1
        num_splits = 2

        with self.assertRaises(ValueError):
            random_region_graph(
                X=random_variables,
                depth=depth,
                replicas=replicas,
                num_splits=num_splits,
            )

    def test_region_graph_not_enough_randomvars(self):
        for num_random_variables in range(1, 3):
            random_variables = set(range(1, num_random_variables + 1))
            depth = 1
            replicas = 1
            num_splits = num_random_variables + 1

            with self.assertRaises(ValueError):
                random_region_graph(
                    X=random_variables,
                    depth=depth,
                    replicas=replicas,
                    num_splits=num_splits,
                )

    def test_region_graph_negative_depth(self):
        random_variables = set(range(1, 3))
        depth = -1
        replicas = 1
        num_splits = 2

        with self.assertRaises(ValueError):
            random_region_graph(
                X=random_variables,
                depth=depth,
                replicas=replicas,
                num_splits=num_splits,
            )

    def test_region_graph_no_replicas(self):
        random_variables = set(range(1, 3))
        depth = 1
        replicas = 0
        num_splits = 2

        with self.assertRaises(ValueError):
            random_region_graph(
                X=random_variables,
                depth=depth,
                replicas=replicas,
                num_splits=num_splits,
            )

    def test_region_graph_invalid_num_split(self):
        random_variables = set(range(1, 3))
        depth = 1
        replicas = 0
        num_splits = 1

        with self.assertRaises(ValueError):
            random_region_graph(
                X=random_variables,
                depth=depth,
                replicas=replicas,
                num_splits=num_splits,
            )

    def test_region_graph_structure_1(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 2
        replicas = 1
        num_splits = 2

        simple_region_graph = random_region_graph(
            X=random_variables, depth=depth, replicas=replicas, num_splits=num_splits
        )

        regions_by_depth, leaves = _get_regions_by_depth(simple_region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 2)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 7)
        self.assertEqual(len(simple_region_graph.partitions), 3)
        self.assertEqual(len(leaves), 4)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )

    def test_region_graph_structure_2(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 1
        num_splits = 2

        simple_region_graph = random_region_graph(
            X=random_variables, depth=depth, replicas=replicas, num_splits=num_splits
        )

        regions_by_depth, leaves = _get_regions_by_depth(simple_region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 3)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 13)
        self.assertEqual(len(simple_region_graph.partitions), 6)
        self.assertEqual(len(leaves), 7)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )

    def test_region_graph_structure_3(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 2
        num_splits = 2

        simple_region_graph = random_region_graph(
            X=random_variables, depth=depth, replicas=replicas, num_splits=num_splits
        )

        regions_by_depth, leaves = _get_regions_by_depth(simple_region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 3)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 25)
        self.assertEqual(len(simple_region_graph.partitions), 12)
        self.assertEqual(len(leaves), 14)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )

    def test_region_graph_structure_4(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 2
        num_splits = 3

        simple_region_graph = random_region_graph(
            X=random_variables, depth=depth, replicas=replicas, num_splits=num_splits
        )

        regions_by_depth, leaves = _get_regions_by_depth(simple_region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 2)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 13)
        self.assertEqual(len(simple_region_graph.partitions), 4)
        self.assertEqual(len(leaves), 10)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )


if __name__ == "__main__":
    unittest.main()
