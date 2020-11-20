"""
=================
Multivariate Leaf
=================

We can learn a SPN with multivariate leaf. This example demonstrates learning
an SPN with Chow Liu tree (CLTs) as multivariate leaves.
"""

import numpy as np

np.random.seed(123)

from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Inference import log_likelihood

train_data = np.random.binomial(1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1], size=(100, 10))


ds_context = Context(
    parametric_types=[
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
        Bernoulli,
    ]
).add_domains(train_data)

spn = learn_parametric(
    train_data,
    ds_context,
    min_instances_slice=20,
    min_features_slice=1,
    multivariate_leaf=True,
    leaves=create_cltree_leaf,
)

ll = log_likelihood(spn, train_data)
print(np.mean(ll))
