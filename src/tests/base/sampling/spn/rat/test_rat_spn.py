from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
from spflow.base.structure.spn.rat.region_graph import random_region_graph
from spflow.base.structure.spn.rat.rat_spn import RatSPN
from spflow.base.inference.spn.rat.rat_spn import log_likelihood
from spflow.base.sampling.spn.rat.rat_spn import sample
from spflow.base.inference.module import log_likelihood
from spflow.base.sampling.module import sample
from spflow.base.inference.nested_module import log_likelihood
from spflow.base.sampling.nested_module import sample
from spflow.base.inference.spn.nodes.sum_node import log_likelihood
from spflow.base.inference.spn.nodes.product_node import log_likelihood
from spflow.base.sampling.spn.nodes.sum_node import sample
from spflow.base.sampling.spn.nodes.product_node import sample
from spflow.base.inference.nodes.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.sampling.nodes.leaves.parametric.gaussian import sample
from spflow.base.inference.spn.layers.sum_layer import log_likelihood
from spflow.base.inference.spn.layers.partition_layer import log_likelihood
from spflow.base.inference.spn.layers.hadamard_layer import log_likelihood
from spflow.base.sampling.spn.layers.sum_layer import sample
from spflow.base.sampling.spn.layers.partition_layer import sample
from spflow.base.sampling.spn.layers.hadamard_layer import sample
from spflow.base.inference.layers.leaves.parametric.gaussian import (
    log_likelihood,
)
from spflow.base.sampling.layers.leaves.parametric.gaussian import sample
import unittest

import numpy as np
import random


class TestModule(unittest.TestCase):
    def test_sampling(self):

        # set seed
        np.random.seed(0)
        random.seed(0)

        # create region graph
        scope = Scope(list(range(128)))
        region_graph = random_region_graph(
            scope, depth=5, replicas=2, n_splits=2
        )
        feature_ctx = FeatureContext(
            scope, {rv: FeatureTypes.Gaussian for rv in scope.query}
        )

        # create torch rat spn from region graph
        rat = RatSPN(
            region_graph,
            feature_ctx,
            n_root_nodes=4,
            n_region_nodes=2,
            n_leaf_nodes=3,
        )

        # since RAT-SPNs are completely composed out of tested layers and nodes, the validity does not have to be checked specifically
        # should simply NOT raise an error
        sample(rat, 100)


if __name__ == "__main__":
    unittest.main()
