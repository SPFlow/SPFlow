"""TODO
"""
from typing import Dict, List, Optional, Tuple, Type, Union

from spflow.modules.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope

# ----- non-conditional modules -----
from spflow.torch.structure.general.layer.leaf.bernoulli import (
    Bernoulli,
    BernoulliLayer,
)
from spflow.torch.structure.general.layer.leaf.binomial import (
    Binomial,
    BinomialLayer,
)

# ----- conditional modules -----
from spflow.torch.structure.general.layer.leaf.cond_bernoulli import (
    CondBernoulli,
    CondBernoulliLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_binomial import (
    CondBinomial,
    CondBinomialLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_exponential import (
    CondExponential,
    CondExponentialLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_gamma import (
    CondGamma,
    CondGammaLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_gaussian import (
    CondGaussian,
    CondGaussianLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_geometric import (
    CondGeometric,
    CondGeometricLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_log_normal import (
    CondLogNormal,
    CondLogNormalLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
    CondMultivariateGaussianLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_negative_binomial import (
    CondNegativeBinomial,
    CondNegativeBinomialLayer,
)
from spflow.torch.structure.general.layer.leaf.cond_poisson import (
    CondPoisson,
    CondPoissonLayer,
)
from spflow.torch.structure.general.layer.leaf.exponential import (
    Exponential,
    ExponentialLayer,
)
from spflow.torch.structure.general.layer.leaf.gamma import (
    Gamma,
    GammaLayer,
)
from spflow.torch.structure.general.layer.leaf.gaussian import (
    Gaussian,
    GaussianLayer,
)
from spflow.torch.structure.general.layer.leaf.geometric import (
    Geometric,
    GeometricLayer,
)
from spflow.torch.structure.general.layer.leaf.hypergeometric import (
    Hypergeometric,
    HypergeometricLayer,
)
from spflow.torch.structure.general.layer.leaf.log_normal import (
    LogNormal,
    LogNormalLayer,
)
from spflow.torch.structure.general.layer.leaf.multivariate_gaussian import (
    MultivariateGaussian,
    MultivariateGaussianLayer,
)
from spflow.torch.structure.general.layer.leaf.negative_binomial import (
    NegativeBinomial,
    NegativeBinomialLayer,
)
from spflow.torch.structure.general.layer.leaf.poisson import (
    Poisson,
    PoissonLayer,
)
from spflow.torch.structure.general.layer.leaf.uniform import (
    Uniform,
    UniformLayer,
)


class AutoLeaf:
    """TODO"""

    __leaf_map: dict[int, Module] = {
        # univariate nodes
        0: Bernoulli,
        1: Binomial,
        2: Exponential,
        3: Gamma,
        4: Gaussian,
        5: Geometric,
        6: Hypergeometric,
        7: LogNormal,
        8: NegativeBinomial,
        9: Poisson,
        10: Uniform,
        11: CondBernoulli,
        12: CondBinomial,
        13: CondExponential,
        14: CondGamma,
        15: CondGaussian,
        16: CondGeometric,
        17: CondLogNormal,
        18: CondNegativeBinomial,
        19: CondPoisson,
        # multivariate nodes (make sure they have lower priority than univariate nodes since they may also match univariate signatures)
        100: MultivariateGaussian,
        101: CondMultivariateGaussian,
        # layers (should come after nodes, since layers can also represent single outputs)
        200: BernoulliLayer,
        201: BinomialLayer,
        202: ExponentialLayer,
        203: GammaLayer,
        204: GaussianLayer,
        205: GeometricLayer,
        206: HypergeometricLayer,
        207: LogNormalLayer,
        208: NegativeBinomialLayer,
        209: PoissonLayer,
        210: UniformLayer,
        211: CondBernoulliLayer,
        212: CondBinomialLayer,
        213: CondExponentialLayer,
        214: CondGammaLayer,
        215: CondGaussianLayer,
        216: CondGeometricLayer,
        217: CondLogNormalLayer,
        218: CondNegativeBinomialLayer,
        219: CondPoissonLayer,
        # multivariate layers (make sure they have lower priority than univariate layers since they may also match univariate signatures)
        300: MultivariateGaussianLayer,
        301: CondMultivariateGaussianLayer,
    }

    def __new__(cls, signatures: list[FeatureContext]):
        """"""
        leaf_type = AutoLeaf.infer(signatures)

        if leaf_type is None:
            raise ValueError("Could not infer leaf type from the following signatures: {signatures}.")

        return leaf_type.from_signatures(signatures)

    @classmethod
    def __push_down(cls, key) -> None:
        """TODO"""
        if key not in cls.__leaf_map.keys():
            return
        if key + 1 in cls.__leaf_map.keys():
            cls.__push_down(key + 1)
        # delete entry under current id
        value = cls.__leaf_map.pop(key)
        cls.__leaf_map[key + 1] = value

    @classmethod
    def __next_key(cls, start: Optional[int] = None) -> id:
        """TODO"""
        if start is None:
            # start from beginning
            key = 0
        else:
            key = start

        # find next best available value
        while key in cls.__leaf_map.keys():
            key += 1

        return key

    @classmethod
    def register(
        cls,
        module: Module,
        priority: Optional[int] = None,
        before: Optional[list[Module]] = None,
        type: str = "node",
        arity: str = "uni",
    ) -> None:
        """TODO"""
        # if module already registered it is registered again at bottom of priority list
        for id, m in list(cls.__leaf_map.items()):
            if module == m:
                del cls.__leaf_map[id]

        if priority is None:
            start = 0

            if type == "node":
                pass
            elif type == "layer":
                start += 200
            else:
                ValueError("TODO.")

            if arity == "uni":
                pass
            elif arity == "multi":
                start += 100
            else:
                ValueError("TODO.")

            priority = cls.__next_key(start)

        if before is None:
            # right beneath largest value
            before = max(cls.__leaf_map.keys()) + 2
        else:
            before_ids = []
            for ref in before:
                if isinstance(ref, int):
                    # reference is already the key
                    before_ids.append(ref)
                else:
                    # reference is a module
                    for k, m in cls.__leaf_map.items():
                        if m == ref:
                            before_ids.append(k)
            # take minimum value as lower bound
            before = min(before_ids) if before_ids else max(cls.__leaf_map.keys()) + 2

        if priority < before:
            # use value preference
            cls.__push_down(priority)
            cls.__leaf_map[priority] = module
        else:
            # take value of lower bound
            cls.__push_down(before)
            cls.__leaf_map[before] = module

    @classmethod
    def is_registered(cls, module) -> bool:
        """TODO"""
        return module in list(cls.__leaf_map.values())

    @classmethod
    def infer(cls, signatures: list[tuple[list[Union[MetaType, FeatureType, type[FeatureType]]], Scope]]) -> Union[Module, None]:  # type: ignore
        """TODO"""
        keys = sorted(list(cls.__leaf_map.keys()))

        for key in keys:
            if cls.__leaf_map[key].accepts(signatures):
                return cls.__leaf_map[key]

        return None
