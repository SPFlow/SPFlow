from spn.base.rat import region_graph as rg
import unittest


class TestRegionGraph(unittest.TestCase):
    def test_region_graph_empty_randomvars(self):
        random_variables = None
        depth = 1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_variables, depth=depth, replicas=replicas)

    def test_region_graph_not_enough_randomvars(self):
        random_varables = set(range(1, 2))
        depth = 1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)

    def test_region_graph_negative_depth(self):
        random_varables = set(range(1, 3))
        depth = -1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)

    def test_region_graph_no_replicas(self):
        random_varables = set(range(1, 3))
        depth = 1
        replicas = 0

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)

    def test_region_graph_X7d2r1(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 2
        replicas = 1

        simple_region_graph = rg.random_region_graph(
            X=random_variables, depth=depth, replicas=replicas
        )

        regions_by_depth, leaves = rg._get_regions_by_depth(simple_region_graph)

        # assert depth of region nodes is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 2)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 7)
        self.assertEqual(len(simple_region_graph.partitions), 3)
        self.assertEqual(len(leaves), 4)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )

    def test_region_graph_X7d3r1(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 1

        simple_region_graph = rg.random_region_graph(
            X=random_variables, depth=depth, replicas=replicas
        )

        regions_by_depth, leaves = rg._get_regions_by_depth(simple_region_graph)

        # assert depth of region nodes is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 3)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 13)
        self.assertEqual(len(simple_region_graph.partitions), 6)
        self.assertEqual(len(leaves), 7)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )

    def test_region_graph_X7d3r2(self):
        random_variables = set(range(1, 8))  # 7 randomvars
        depth = 3
        replicas = 2

        simple_region_graph = rg.random_region_graph(
            X=random_variables, depth=depth, replicas=replicas
        )

        regions_by_depth, leaves = rg._get_regions_by_depth(simple_region_graph)

        # assert depth of region nodes is equal to argument depth
        self.assertEqual(len(regions_by_depth) - 1, 3)
        # assert correct number of regions, partitions, and leaves
        self.assertEqual(len(simple_region_graph.regions), 25)
        self.assertEqual(len(simple_region_graph.partitions), 12)
        self.assertEqual(len(leaves), 14)
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(
            random_variables, set.union(*[region.random_variables for region in leaves])
        )


if __name__ == "__main__":
    unittest.main()
