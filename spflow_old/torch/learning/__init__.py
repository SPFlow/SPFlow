# ---- sub-packages -----

# ---- specific imports
from .expectation_maximization import expectation_maximization
from .general.layer.leaf.bernoulli import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.binomial import em, maximum_likelihood_estimation
from .general.layer.leaf.exponential import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.gamma import em, maximum_likelihood_estimation
from .general.layer.leaf.gaussian import em, maximum_likelihood_estimation
from .general.layer.leaf.geometric import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.hypergeometric import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.log_normal import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.multivariate_gaussian import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.negative_binomial import (
    em,
    maximum_likelihood_estimation,
)
from .general.layer.leaf.poisson import em, maximum_likelihood_estimation
from .general.layer.leaf.uniform import em, maximum_likelihood_estimation

# import all definitions of 'maximum_likelihood_estimation' and 'em'
from .general.node.leaf.bernoulli import em, maximum_likelihood_estimation
from .general.node.leaf.binomial import em, maximum_likelihood_estimation
from .general.node.leaf.exponential import (
    em,
    maximum_likelihood_estimation,
)
from .general.node.leaf.gamma import em, maximum_likelihood_estimation
from .general.node.leaf.gaussian import em, maximum_likelihood_estimation
from .general.node.leaf.geometric import em, maximum_likelihood_estimation
from .general.node.leaf.hypergeometric import (
    em,
    maximum_likelihood_estimation,
)
from .general.node.leaf.log_normal import (
    em,
    maximum_likelihood_estimation,
)
from .general.node.leaf.multivariate_gaussian import (
    em,
    maximum_likelihood_estimation,
)
from .general.node.leaf.negative_binomial import (
    em,
    maximum_likelihood_estimation,
)
from .general.node.leaf.poisson import em, maximum_likelihood_estimation
from .general.node.leaf.uniform import em, maximum_likelihood_estimation
