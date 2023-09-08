from typing import List, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class BinomialLayer:
    def __new__(cls, scope: Union[Scope, List[Scope]], n: Union[int, List[int], T], p: Union[int, float, List[float], T] = 0.5, n_nodes: int = 1,**kwargs):
        from spflow.base.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TensorlyBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TorchBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes,**kwargs)
        elif backend == "pytorch":
            return TorchBinomial(scope=scope, n=n, p=p, n_nodes=n_nodes,**kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TensorlyBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TorchBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchBinomial.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TensorlyBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.binomial import BinomialLayer as TorchBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

