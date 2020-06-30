"""
=======================
Cutset Networks (CNets)
=======================

With SPFlow we can learn both the structure and the parameters of CNets, a
particular kind of SPNs with CLTs as leaf providing exact MPE inference.
"""

import numpy as np

np.random.seed(123)


from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli
from spn.algorithms.LearningWrappers import learn_parametric, learn_cnet
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe

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

# Learning a CNet with a naive mle conditioning
cnet_naive_mle = learn_cnet(train_data, ds_context, cond="naive_mle", min_instances_slice=20, min_features_slice=1)

# Learning a CNet with random conditioning
cnet_random = learn_cnet(train_data, ds_context, cond="random", min_instances_slice=20, min_features_slice=1)

ll = log_likelihood(cnet_naive_mle, train_data)
print("Naive mle conditioning", np.mean(ll))

ll = log_likelihood(cnet_random, train_data)
print("Random conditioning", np.mean(ll))

# computing exact MPE
train_data_mpe = train_data.astype(float)
train_data_mpe[:, 0] = np.nan
print(mpe(cnet_random, train_data_mpe))
