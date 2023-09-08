from typing import List, Optional, Callable, Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T
from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondBinomialLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        n: Union[int, List[int], T],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.base.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TensorlyCondBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TorchCondBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TensorlyCondBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TorchCondBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TensorlyCondBinomial
        from spflow.torch.structure.general.layers.leaves.parametric.cond_binomial import CondBinomialLayer as TorchCondBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
