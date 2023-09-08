"""TODO
"""
from typing import Dict, List, Optional, Tuple, Type, Union

# ----- non-conditional modules -----
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
#    Bernoulli,
    BernoulliLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Bernoulli
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    BinomialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Binomial
# ----- conditional modules -----
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondBernoulliLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondBernoulli
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondBinomialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondBinomial
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondExponentialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondExponential
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondGammaLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondGamma
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondGaussianLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondGaussian
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondGeometricLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondGeometric
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondLogNormalLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondLogNormal
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondMultivariateGaussianLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondMultivariateGaussian
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondNegativeBinomialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondNegativeBinomial
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    CondPoissonLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import CondPoisson
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    ExponentialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Exponential
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    GammaLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Gamma
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    GaussianLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Gaussian
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    GeometricLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Geometric
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    HypergeometricLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Hypergeometric
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    LogNormalLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import LogNormal
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    MultivariateGaussianLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import MultivariateGaussian
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    NegativeBinomialLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import NegativeBinomial
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    PoissonLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Poisson
from spflow.tensorly.structure.spn.layers.leaves.parametric import (
    UniformLayer,
)
from spflow.tensorly.structure.spn.nodes.leaves.parametric import Uniform

from spflow.tensorly.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext
from spflow.meta.data.feature_types import FeatureType
from spflow.meta.data.meta_type import MetaType
from spflow.meta.data.scope import Scope


class AutoLeaf:
    """TODO"""

    __leaf_map: Dict[int, Module] = {
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
        301: CondMultivariateGaussianLayer
    }

    def __new__(cls, signatures: List[FeatureContext]):
        """TODO"""
        leaf_type = AutoLeaf.infer(signatures)

        if leaf_type is None:
            raise ValueError(f"Could not infer leaf type from the following signatures: {signatures}.")

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
        self,
        module: Module,
        priority: Optional[int] = None,
        before: Optional[List[Module]] = None,
        type: str = "node",
        arity: str = "uni",
    ) -> None:
        """TODO"""
        # if module already registered it is registered again at bottom of priority list
        for id, m in list(self.__leaf_map.items()):
            if module == m:
                del self.__leaf_map[id]

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

            priority = self.__next_key(start)

        if before is None:
            # right beneath largest value
            before = max(self.__leaf_map.keys()) + 2
        else:
            before_ids = []
            for ref in before:
                if isinstance(ref, int):
                    # reference is already the key
                    before_ids.append(ref)
                else:
                    # reference is a module
                    for k, m in self.__leaf_map.items():
                        if m == ref:
                            before_ids.append(k)
            # take minimum value as lower bound
            before = min(before_ids) if before_ids else max(self.__leaf_map.keys()) + 2

        if priority < before:
            # use value preference
            self.__push_down(priority)
            self.__leaf_map[priority] = module
        else:
            # take value of lower bound
            self.__push_down(before)
            self.__leaf_map[before] = module

    @classmethod
    def is_registered(cls, module) -> bool:
        """TODO"""
        return module in list(cls.__leaf_map.values())

    @classmethod
    def infer(cls, signatures: List[FeatureContext]) -> Union[Module, None]:  # type: ignore
        """TODO"""
        keys = sorted(list(cls.__leaf_map.keys()))

        for key in keys:
            if cls.__leaf_map[key].accepts(signatures):
                return cls.__leaf_map[key]

        return None
