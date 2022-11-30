# ---- sub-packages -----
from . import spn, general  # isort: skip

from .autoleaf import AutoLeaf
from .general.layers.leaves.parametric.bernoulli import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.binomial import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.cond_bernoulli import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_binomial import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.cond_exponential import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_gamma import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.cond_gaussian import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.cond_geometric import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_log_normal import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_multivariate_gaussian import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_negative_binomial import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.cond_poisson import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.exponential import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.gamma import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.gaussian import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.geometric import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.hypergeometric import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.log_normal import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.multivariate_gaussian import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.negative_binomial import (
    marginalize,
    toBase,
    toTorch,
)
from .general.layers.leaves.parametric.poisson import marginalize, toBase, toTorch
from .general.layers.leaves.parametric.uniform import marginalize, toBase, toTorch
from .general.nodes.leaf_node import LeafNode

# import all definitions of 'marginalize', 'toBase' and 'toTorch'
from .general.nodes.node import marginalize  # handles all leaf nodes

# ---- specific imports -----
from .module import Module
from .nested_module import NestedModule
from .spn.layers.cond_sum_layer import marginalize, toBase, toTorch
from .spn.layers.hadamard_layer import marginalize, toBase, toTorch
from .spn.layers.partition_layer import marginalize, toBase, toTorch
from .spn.layers.product_layer import marginalize, toBase, toTorch
from .spn.layers.sum_layer import marginalize, toBase, toTorch
from .spn.nodes.cond_sum_node import marginalize, toBase, toTorch
from .spn.nodes.product_node import marginalize, toBase, toTorch
from .spn.nodes.sum_node import marginalize, toBase, toTorch
from .spn.rat.rat_spn import marginalize, toBase, toTorch
