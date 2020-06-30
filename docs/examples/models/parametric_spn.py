"""
==============
Parametric SPN
==============

The setup is similar to the Mixed SPN, but here we learn a parametric SPN.
"""

import numpy as np

np.random.seed(123)

from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

a = np.random.randint(2, size=1000).reshape(-1, 1)
b = np.random.randint(3, size=1000).reshape(-1, 1)
c = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]
d = 5 * a + 3 * b + c

train_data = np.c_[a, b, c, d]

ds_context = Context(parametric_types=[Categorical, Categorical, Gaussian, Gaussian]).add_domains(train_data)

spn = learn_parametric(train_data, ds_context, min_instances_slice=20)
