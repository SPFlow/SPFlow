import numpy as np
from pydot import frozendict


class RegionGraph(object):
    """Represents a region graph.

    Gets initialized with a set of 'items', where usually items is range(0,D), where D is the number of dimensions.
    Regions are represented as subsets of items, implemented as pyhon's frozensets. The set of all regions is stored in
    regions.
    Partitions are sets of regions, implemented as frozensets of (non-overlapping) frozensets. The set of all partitions
    is stored in _partitions.
    For each region we have a list of their child partitions, held in dictionary _child_partitions.
    """

    def __init__(self, items, seed = 12345):

        self._items = frozenset(items)

        # Regions
        self._regions = set()
        self._child_partitions = dict()

        # Partitions
        self._partitions = set()

        # Private random generator
        self.rand_state = np.random.RandomState(seed)

        # The root region (== _items) is already part of the region graph
        self._regions.add(self._items)

        self._layers = []

    def get_child_partitions(self):
        return frozendict(self._child_partitions)

    def get_num_items(self):
        """Return the number of items"""

        return len(self._items)

    def get_root_region(self):
        """Get root region."""

        return self._items

    def get_region(self, region):
        """Get a region and create if it does not exist."""

        region = frozenset(region)

        if not region <= self._items:
            raise ValueError('Argument region is not a sub-set of _items.')

        self._regions.add(region)
        return region

    def merge(self, regions):
        """Merge a set of regions.

        Regions is an iterable collection of non-overlapping regions (i.e. sub-frozensets of _items).
        This method merges these regions to a super-region, and introduces this super-region in the region graph.
        Furthermore it introduces the corresponding partition in the region graph.
        """

        super_region = set()
        num_elems_in_subregions = 0
        for x in regions:
            if x not in self._regions:
                raise ValueError('Trying to merge non-existing regions.')
            num_elems_in_subregions += len(x)
            super_region.update(x)

        if num_elems_in_subregions != len(super_region):
            raise ValueError('Sub-regions overlapping.')

        partition = frozenset(regions)
        super_region = self.get_region(super_region)
        if partition not in self._partitions:
            self._partitions.add(partition)
            super_region_children = self._child_partitions.get(super_region, [])
            self._child_partitions[super_region] = super_region_children + [partition]

        return super_region

    def get_random_atomic_regions(self, n, root_region = None):
        """Split the item set into atomic regions of length n.

        Generate a set of regions of length n (or the number of remaining elements), and introduce them in the
        region graph.
        """

        if root_region:
            rand_item_list = list(self.rand_state.permutation(list(root_region)))
        else:
            rand_item_list = list(self.rand_state.permutation(list(self._items)))
        return [self.get_region(rand_item_list[k:k + n]) for k in range(0, len(rand_item_list), n)]

    def random_binary_trees(self, m, k, n, root_region = None):
        """Split items into regions of size n and grow k times a random binary merge tree. Repeat m times.

        repeat m times:
            get random atomic regions of size n
            repeat k times:
                repeatedly select two regions and merge them;
                introduce the merged region in the region graph
        """

        for _ in xrange(0, m):
            atomic_orig = self.get_random_atomic_regions(n, root_region)
            for _ in xrange(0, k):
                atomic = list(atomic_orig)
                while len(atomic) > 1:
                    region1 = atomic.pop(self.rand_state.randint(len(atomic)))
                    region2 = atomic.pop(self.rand_state.randint(len(atomic)))
                    atomic.append(self.merge([region1, region2]))

    def get_atomic_regions(self):
        """Get atomic regions, i.e. regions which don't have child partitions."""

        return [x for x in self._regions if x not in self._child_partitions]

    def random_split(self, num_parts, num_recursions=1, region=None):
        """Split a region in n random parts and introduce the corresponding partition in the region graph."""

        if num_recursions < 1:
            return None

        if not region:
            region = self._items

        if region not in self._regions:
            raise LookupError('Trying to split non-existing region.')

        if len(region) == 1:
            return None

        region_list = list(self.rand_state.permutation(list(region)))

        num_parts = min(len(region_list), num_parts)
        q = len(region_list) // num_parts
        r = len(region_list) % num_parts

        partition = []
        idx = 0
        for k in range(0, num_parts):
            inc = q + 1 if k < r else q
            sub_region = frozenset(region_list[idx:idx+inc])
            partition.append(sub_region)
            self._regions.add(sub_region)
            idx = idx + inc

        partition = frozenset(partition)
        self._partitions.add(partition)
        region_children = self._child_partitions.get(region, [])
        self._child_partitions[region] = region_children + [partition]

        if num_recursions > 1:
            for r in partition:
                self.random_split(num_parts, num_recursions-1, r)

        return partition

    def imbalanced_split(self, num_left=1, num_recursions=1, region=None):
        """..."""

        if num_recursions < 1:
            return None

        if not region:
            region = self._items

        if region not in self._regions:
            raise LookupError('Trying to split non-existing region.')

        if len(region) == 1:
            return None

        if len(region) <= num_left:
            return None

        region_list = list(self.rand_state.permutation(list(region)))

        partition = []

        left_region = frozenset(region_list[0:num_left])
        partition.append(left_region)
        self._regions.add(left_region)

        right_region = frozenset(region_list[num_left:])
        partition.append(right_region)
        self._regions.add(right_region)

        partition = frozenset(partition)
        self._partitions.add(partition)
        region_children = self._child_partitions.get(region, [])
        self._child_partitions[region] = region_children + [partition]

        if num_recursions > 1:
            self.imbalanced_split(num_left, num_recursions - 1, right_region)

        return partition

    def make_layers(self):
        """Make a layered structure.

        _layer[0] will contain atomic regions
        _layer[k], when k is odd, will contain partitions
        _layer[k], when k is even, will contain regions
        """

        seen_regions = set()
        seen_partitions = set()

        atomic_regions = self.get_atomic_regions()
        self._layers = [atomic_regions]
        if (len(atomic_regions) == 1) and (self._items in atomic_regions):
            return

        while len(seen_regions) != len(self._regions) or len(seen_partitions) != len(self._partitions):
            seen_regions.update(atomic_regions)

            # the next partition layer contains all partitions which have not been visited (seen)
            # and all its child regions have been visited
            next_partition_layer = [p for p in self._partitions if p not in seen_partitions
                                    and all([r in seen_regions for r in p])]
            self._layers.append(next_partition_layer)
            seen_partitions.update(next_partition_layer)

            # similar as above, but now for regions
            next_region_layer = [r for r in self._regions if r not in seen_regions
                                 and all([p in seen_partitions for p in self._child_partitions[r]])]
            self._layers.append(next_region_layer)
            seen_regions.update(next_region_layer)

        return self._layers

