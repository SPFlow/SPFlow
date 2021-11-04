"""
============================
Tractable Marginal Inference
============================

This shows how to perform marginal inference on Sum-Product Networks.

This picks up from the :ref:`marginalizing_an_spn` tutorial.
"""

from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.Marginalization import marginalize

from spn.io.Graphics import draw_spn
import matplotlib.pyplot as plt

p0 = Product(children=[Categorical(p=[0.3, 0.7], scope=1), Categorical(p=[0.4, 0.6], scope=2)])
p1 = Product(children=[Categorical(p=[0.5, 0.5], scope=1), Categorical(p=[0.6, 0.4], scope=2)])
s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), s1])
p3 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), Categorical(p=[0.3, 0.7], scope=1)])
p4 = Product(children=[p3, Categorical(p=[0.4, 0.6], scope=2)])
spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

assign_ids(spn)
rebuild_scopes_bottom_up(spn)

spn_marg = marginalize(spn, [1, 2])

# %%
# Here is an example on how to evaluate the SPNs from the
# :ref:`marginalizing_an_spn` tutorial.
# Since we have 3 variables, we want to create a 2D numpy array of 3 columns
# and 1 row.

import numpy as np
test_data = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)

# %%
# We can then compute the log-likelihood:

from spn.algorithms.Inference import log_likelihood

ll = log_likelihood(spn, test_data)
print(ll, np.exp(ll))

# %%
# We can also compute the log-likelihood of the marginal SPN.
# Notice we use the same ``test_data`` as input: the SPN still expects an array
# with data at columns 1 and 2, but ignores column 0.

llm = log_likelihood(spn_marg, test_data)
print(llm, np.exp(llm))

# %%
# Another alternative would be to perform **marginal inference** on the original
# SPN. This can be done by setting ``np.nan`` for the feature we want to
# marginalize on the fly. It does not change the structure.

test_data2 = np.array([np.nan, 0.0, 1.0]).reshape(-1, 3)
llom = log_likelihood(spn, test_data2)
print(llom, np.exp(llom))

# %%
# Observe that the *marginal inference solution* and the *marginal SPN solution*
# are the same.

print(np.array_equal(llm, llom))
