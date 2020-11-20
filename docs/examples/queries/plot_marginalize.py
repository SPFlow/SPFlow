"""
.. _marginalizing_an_spn:

====================
Marginalizing an SPN
====================

"Marginalizing an SPN" means summing out all other non-relevant variables.
"""

# sphinx_gallery_thumbnail_number = 2

from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up

from spn.io.Graphics import draw_spn
import matplotlib.pyplot as plt

# %%
# We will start with the Sum-Product Network structure from the
# :ref:`composing_spn_object_hierarchy` example.

p0 = Product(children=[Categorical(p=[0.3, 0.7], scope=1), Categorical(p=[0.4, 0.6], scope=2)])
p1 = Product(children=[Categorical(p=[0.5, 0.5], scope=1), Categorical(p=[0.6, 0.4], scope=2)])
s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), s1])
p3 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), Categorical(p=[0.3, 0.7], scope=1)])
p4 = Product(children=[p3, Categorical(p=[0.4, 0.6], scope=2)])
spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

assign_ids(spn)
rebuild_scopes_bottom_up(spn)

ax = draw_spn(spn)

# %%
# If we want to marginalize this SPN by summing out all other variables
# to leave variables 1 and 2, we can do this as follows:

from spn.algorithms.Marginalization import marginalize

spn_marg = marginalize(spn, [1, 2])

# %%
# This marginalizes all the variables *not* in :math:`[1, 2]`, and create a
# *new* structure that knows nothing about the previous one nor about the
# variable 0.

draw_spn(spn_marg)
