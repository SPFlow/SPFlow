"""
=========
Mixed SPN
=========

This demonstrates learning a Mixed Sum-Product Network (MSPN) where the data is
composed of variables drawn from multiple types of distributions.
"""

import numpy as np

np.random.seed(123)

from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType

##########################################################################
# We will compose a data set from four features:
#
# - two Discrete
# - two Real valued
#
# A and B are discrete, C and D are real-valued.

a = np.random.randint(2, size=1000).reshape(-1, 1)
b = np.random.randint(3, size=1000).reshape(-1, 1)
c = np.r_[np.random.normal(10, 5, (300, 1)), np.random.normal(20, 10, (700, 1))]
d = 5 * a + 3 * b + c

train_data = np.c_[a, b, c, d]

##########################################################################
# The types of distributions are known ahead of time, so we can add these
# as ``meta_types`` in the ``Context``:

ds_context = Context(meta_types=[MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL, MetaType.REAL])
ds_context.add_domains(train_data)

##########################################################################
# Finally, we learn the MSPN:

mspn = learn_mspn(train_data, ds_context, min_instances_slice=20)
