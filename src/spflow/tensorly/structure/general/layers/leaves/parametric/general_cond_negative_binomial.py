from typing import List, Optional, Callable,Union

import tensorly as tl
from spflow.tensorly.utils.helper_functions import T

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondNegativeBinomialLayer:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Union[Scope, List[Scope]],
        n: Union[int, List[int], T],
        cond_f: Optional[Union[Callable, List[Callable]]] = None,
        n_nodes: int = 1,
        **kwargs):
        from spflow.tensorly.structure.general.layers.leaves import CondNegativeBinomialLayer as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.layers.leaves import CondNegativeBinomialLayer as TorchCondNegativeBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f, n_nodes=n_nodes, **kwargs)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.layers.leaves import CondNegativeBinomialLayer as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.layers.leaves import CondNegativeBinomialLayer as TorchCondNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.layers.leaves import CondNegativeBinomialLayer as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.layers.leaves import CondNegativeBinomialLayer as TorchCondNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")
