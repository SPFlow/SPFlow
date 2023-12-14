"""TODO
"""
from typing import Dict, List, Optional, Union

# ----- non-conditional modules -----
from spflow.tensorly.structure.spn.layer.leaf import (
#    Bernoulli,
    BernoulliLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Bernoulli
from spflow.tensorly.structure.spn.layer.leaf import (
    BinomialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Binomial
# ----- conditional modules -----
from spflow.tensorly.structure.spn.layer.leaf import (
    CondBernoulliLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondBernoulli
from spflow.tensorly.structure.spn.layer.leaf import (
    CondBinomialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondBinomial
from spflow.tensorly.structure.spn.layer.leaf import (
    CondExponentialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondExponential
from spflow.tensorly.structure.spn.layer.leaf import (
    CondGammaLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondGamma
from spflow.tensorly.structure.spn.layer.leaf import (
    CondGaussianLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondGaussian
from spflow.tensorly.structure.spn.layer.leaf import (
    CondGeometricLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondGeometric
from spflow.tensorly.structure.spn.layer.leaf import (
    CondLogNormalLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondLogNormal
from spflow.tensorly.structure.spn.layer.leaf import (
    CondMultivariateGaussianLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondMultivariateGaussian
from spflow.tensorly.structure.spn.layer.leaf import (
    CondNegativeBinomialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondNegativeBinomial
from spflow.tensorly.structure.spn.layer.leaf import (
    CondPoissonLayer,
)
from spflow.tensorly.structure.spn.node.leaf import CondPoisson
from spflow.tensorly.structure.spn.layer.leaf import (
    ExponentialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Exponential
from spflow.tensorly.structure.spn.layer.leaf import (
    GammaLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Gamma
from spflow.tensorly.structure.spn.layer.leaf import (
    GaussianLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Gaussian
from spflow.tensorly.structure.spn.layer.leaf import (
    GeometricLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Geometric
from spflow.tensorly.structure.spn.layer.leaf import (
    HypergeometricLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Hypergeometric
from spflow.tensorly.structure.spn.layer.leaf import (
    LogNormalLayer,
)
from spflow.tensorly.structure.spn.node.leaf import LogNormal
from spflow.tensorly.structure.spn.layer.leaf import (
    MultivariateGaussianLayer,
)
from spflow.tensorly.structure.spn.node.leaf import MultivariateGaussian
from spflow.tensorly.structure.spn.layer.leaf import (
    NegativeBinomialLayer,
)
from spflow.tensorly.structure.spn.node.leaf import NegativeBinomial
from spflow.tensorly.structure.spn.layer.leaf import (
    PoissonLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Poisson
from spflow.tensorly.structure.spn.layer.leaf import (
    UniformLayer,
)
from spflow.tensorly.structure.spn.node.leaf import Uniform

from spflow.meta.structure.module import Module
from spflow.meta.data.feature_context import FeatureContext


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
