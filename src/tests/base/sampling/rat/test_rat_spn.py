from spflow.meta.scope.scope import Scope
from spflow.base.structure.rat.region_graph import random_region_graph
from spflow.base.structure.rat.rat_spn import RatSPN
from spflow.base.inference.rat.rat_spn import log_likelihood
from spflow.base.sampling.rat.rat_spn import sample
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample
from spflow.base.inference.nodes.node import log_likelihood
from spflow.base.sampling.nodes.node import sample
from spflow.base.inference.nodes.leaves.parametric.gaussian import log_likelihood
from spflow.base.sampling.nodes.leaves.parametric.gaussian import sample
from spflow.base.inference.layers.layer import log_likelihood
from spflow.base.sampling.layers.layer import sample
from spflow.base.inference.layers.leaves.parametric.gaussian import log_likelihood
from spflow.base.sampling.layers.leaves.parametric.gaussian import sample
import unittest


class TestModule(unittest.TestCase):    
    def test_sampling(self):
        # create region graph
        region_graph = random_region_graph(scope=Scope(list(range(128))), depth=5, replicas=2, n_splits=2)

        # create torch rat spn from region graph
        rat = RatSPN(region_graph, n_root_nodes=4, n_region_nodes=2, n_leaf_nodes=3)

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        sample(rat, 100)


if __name__ == "__main__":
    unittest.main()