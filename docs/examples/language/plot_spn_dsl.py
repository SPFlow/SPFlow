"""
=================================
Domain Specific Language for SPNs
=================================

We start by creating an SPN. Using a Domain-Specific Language (DSL), we can
quickly create an SPN of categorical leave nodes like this:
"""

from spn.structure.leaves.parametric.Parametric import Categorical
from spn.io.Graphics import draw_spn

import matplotlib.pyplot as plt


spn = 0.4 * (
    Categorical(p=[0.2, 0.8], scope=0)
    * (
        0.3 * (Categorical(p=[0.3, 0.7], scope=1) * Categorical(p=[0.4, 0.6], scope=2))
        + 0.7 * (Categorical(p=[0.5, 0.5], scope=1) * Categorical(p=[0.6, 0.4], scope=2))
    )
) + 0.6 * (Categorical(p=[0.2, 0.8], scope=0) * Categorical(p=[0.3, 0.7], scope=1) * Categorical(p=[0.4, 0.6], scope=2))

ax = draw_spn(spn)
