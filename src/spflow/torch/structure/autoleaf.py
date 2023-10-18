"""TODO
"""
from typing import Dict, List, Optional, Tuple, Type, Union

from spflow.base.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope

# ----- non-conditional modules -----
from spflow.torch.structure.general.layers.leaves.parametric.bernoulli import (
    Bernoulli,
    BernoulliLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.binomial import (
    Binomial,
    BinomialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.categorical import Categorical, CategoricalLayer

# ----- conditional modules -----
from spflow.torch.structure.general.layers.leaves.parametric.cond_bernoulli import (
    CondBernoulli,
    CondBernoulliLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_binomial import (
    CondBinomial,
    CondBinomialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_categorical import CondCategorical, CondCategoricalLayer
from spflow.torch.structure.general.layers.leaves.parametric.cond_exponential import (
    CondExponential,
    CondExponentialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_gamma import (
    CondGamma,
    CondGammaLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_gaussian import (
    CondGaussian,
    CondGaussianLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_geometric import (
    CondGeometric,
    CondGeometricLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_log_normal import (
    CondLogNormal,
    CondLogNormalLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_multivariate_gaussian import (
    CondMultivariateGaussian,
    CondMultivariateGaussianLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_negative_binomial import (
    CondNegativeBinomial,
    CondNegativeBinomialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.cond_poisson import (
    CondPoisson,
    CondPoissonLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.exponential import (
    Exponential,
    ExponentialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.gamma import (
    Gamma,
    GammaLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.gaussian import (
    Gaussian,
    GaussianLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.geometric import (
    Geometric,
    GeometricLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.hypergeometric import (
    Hypergeometric,
    HypergeometricLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.log_normal import (
    LogNormal,
    LogNormalLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.multivariate_gaussian import (
    MultivariateGaussian,
    MultivariateGaussianLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.negative_binomial import (
    NegativeBinomial,
    NegativeBinomialLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.poisson import (
    Poisson,
    PoissonLayer,
)
from spflow.torch.structure.general.layers.leaves.parametric.uniform import (
    Uniform,
    UniformLayer,
)


class AutoLeaf:
    """TODO"""

    __leaf_map: Dict[int, Module] = {
        # univariate nodes
        0: Bernoulli,
        1: Binomial,
        2: Categorical,
        3: Exponential,
        4: Gamma,
        5: Gaussian,
        6: Geometric,
        7: Hypergeometric,
        8: LogNormal,
        9: NegativeBinomial,
        10: Poisson,
        11: Uniform,
        12: CondBernoulli,
        13: CondBinomial,
        14: CondCategorical,
        15: CondExponential,
        16: CondGamma,
        17: CondGaussian,
        18: CondGeometric,
        19: CondLogNormal,
        20: CondNegativeBinomial,
        21: CondPoisson,
        # multivariate nodes (make sure they have lower priority than univariate nodes since they may also match univariate signatures)
        100: MultivariateGaussian,
        101: CondMultivariateGaussian,
        # layers (should come after nodes, since layers can also represent single outputs)
        200: BernoulliLayer,
        201: BinomialLayer,
        202: CategoricalLayer,
        203: ExponentialLayer,
        204: GammaLayer,
        205: GaussianLayer,
        206: GeometricLayer,
        207: HypergeometricLayer,
        208: LogNormalLayer,
        209: NegativeBinomialLayer,
        210: PoissonLayer,
        211: UniformLayer,
        212: CondBernoulliLayer,
        213: CondBinomialLayer,
        214: CondCategoricalLayer,
        215: CondExponentialLayer,
        216: CondGammaLayer,
        217: CondGaussianLayer,
        218: CondGeometricLayer,
        219: CondLogNormalLayer,
        220: CondNegativeBinomialLayer,
        221: CondPoissonLayer,
        # multivariate layers (make sure they have lower priority than univariate layers since they may also match univariate signatures)
        300: MultivariateGaussianLayer,
        301: CondMultivariateGaussianLayer,
    }

    def __new__(cls, signatures: List[FeatureContext]):
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
        before: Optional[List[Module]] = None,
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
    def infer(cls, signatures: List[Tuple[List[Union[MetaType, FeatureType, Type[FeatureType]]], Scope]]) -> Union[Module, None]:  # type: ignore
        """TODO"""
        keys = sorted(list(cls.__leaf_map.keys()))

        for key in keys:
            if cls.__leaf_map[key].accepts(signatures):
                return cls.__leaf_map[key]

        return None
