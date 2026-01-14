"""Paper Zoo: Continuous Mixtures of Tractable Probabilistic Models.

This module provides RQMC-based learning for factorized and CLT continuous mixtures,
including latent optimization and compilation to discrete mixtures.

Reference:
    Correia et al., "Continuous Mixtures of Tractable Probabilistic Models" (2023)
"""

from .continuous_mixtures import (
    LatentOptimizationConfig,
    learn_continuous_mixture_cltree,
    learn_continuous_mixture_factorized,
)
from .joint import JointLogLikelihood
from .rqmc import RqmcPoints, rqmc_sobol_normal
