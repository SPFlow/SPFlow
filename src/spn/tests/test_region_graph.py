from typing import Any, List, Set, Tuple
from spn.structure.graph import region_graph as rg
import unittest



class TestRegionGraph(unittest.TestCase):

    def test_simple_region_graph(self):
        random_variables: Set[int] = set(range(1, 8)) # 7 randomvars
        depth: int = 3
        replicas: int = 1

        simple_region_graph = rg.random_region_graph(X=random_variables, depth=depth, replicas=replicas)

        regions_by_depth, leaves = _get_all_regions_by_depth(simple_region_graph)

        # assert depth of region graph is equal to argument depth
        self.assertEqual(len(regions_by_depth)-1, depth)
        # assert that the total number of leafs in the RegionGraph equals to the number of given random_variables times the number of replicas
        self.assertEqual(len(leaves), replicas*len(random_variables))
        # assert that the union of all leafs is equal to the set of random_variables
        self.assertEqual(random_variables, set.union(*[region.random_variables for region in leaves]))


    def test_region_graph_empty_randomvars(self):
        random_variables = None
        depth = 1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_variables, depth=depth, replicas=replicas)
            

    def test_region_graph_not_enough_randomvars(self):
        random_varables: Set[int] = set(range(1, 2))
        depth = 1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)

    
    def test_region_graph_negative_depth(self):
        random_varables: Set[int] = set(range(1, 3))
        depth = -1
        replicas = 1

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)


    def test_region_graph_not_enough_replica(self):
        random_varables: Set[int] = set(range(1, 3))
        depth = 1
        replicas = 0

        with self.assertRaises(ValueError):
            rg.random_region_graph(X=random_varables, depth=depth, replicas=replicas)


def _get_all_regions_by_depth(region_graph: rg.RegionGraph) -> Tuple[List[List[rg.Region]], List[rg.Region]]:
    """
        Returns:
            A 2-dimensional List of Regions, where the first index is the depth in the RegionGraph, and 
            the second index points to the List of Regions of that depth; and a List of all leaf-Regions.
            example A: RegionGraph(X=[1, 2, 3], depth=2, replicas=1):
            ([
                [[1, 2, 3]],
                [[1, 2], [3]], 
                [[1], [2]] 
            ], [
                [1], 
                [2], 
                [3]
            ])

            example B: RegionGraph(X=[1, 2, 3, 4, 5, 6, 7], depth=2, replicas=1):
            ([
                [[1, 2, 3, 4, 5, 6, 7]], 
                [[1, 3, 5, 7], [2, 4, 6]], 
                [[1, 7], [3, 5], [2], [4, 6]]
            ], [
                [1, 7], 
                [3, 5], 
                [2], 
                [4, 6]
            ])

            A 
    """
    depth = 0 
    regions_by_depth = [[region_graph.root_region]]
    leaves = []

    nodes: List[Any] = list(region_graph.root_region.partitions)
    
    while nodes:
        nodeCount = len(nodes)
        
        # increase depth by one for every Region-"layer" encountered
        peek = nodes[0]
        if type(peek) is rg.Region:
            depth += 1
            regions_by_depth.append([])

        while (nodeCount > 0):
            # process the whole "layer" of Regions/Partitions in the nodes-list
            node: Any = nodes.pop(0)
            nodeCount -= 1

            if type(node) is rg.Partition:
                nodes.extend(node.regions)
            elif type(node) is rg.Region:
                regions_by_depth[depth].append(node)
                if node.partitions:
                    nodes.extend(node.partitions)
                else:
                    # if the Region has no children, it is a leaf
                    leaves.append(node)
            else:
                raise ValueError("Node must be Region or Partition")

    return regions_by_depth, leaves    


if __name__ == "__main__":
    unittest.main()
                