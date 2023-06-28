from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondNegativeBinomial:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, n: int, cond_f: Optional[Callable] = None):
        from spflow.tensorly.structure.general.nodes.leaves import CondNegativeBinomial as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves import CondNegativeBinomial as TorchCondNegativeBinomial
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial(scope=scope, n=n, cond_f=cond_f)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves import CondNegativeBinomial as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves import CondNegativeBinomial as TorchCondNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.accepts(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves import CondNegativeBinomial as TensorlyCondNegativeBinomial
        from spflow.torch.structure.general.nodes.leaves import CondNegativeBinomial as TorchCondNegativeBinomial
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondNegativeBinomial.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondNegativeBinomial.from_signatures(signatures)
        else:
            raise NotImplementedError("CondNegativeBinomial is not implemented for this backend")
