from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondPoisson:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.base.structure.general.node.leaf.cond_poisson import CondPoisson as TensorlyCondPoisson
        from spflow.torch.structure.general.node.leaf.cond_poisson import CondPoisson as TorchCondPoisson
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondPoisson(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.base.structure.general.node.leaf.cond_poisson import CondPoisson as TensorlyCondPoisson
        from spflow.torch.structure.general.node.leaf.cond_poisson import CondPoisson as TorchCondPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.accepts(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.base.structure.general.node.leaf.cond_poisson import CondPoisson as TensorlyCondPoisson
        from spflow.torch.structure.general.node.leaf.cond_poisson import CondPoisson as TorchCondPoisson
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondPoisson.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondPoisson.from_signatures(signatures)
        else:
            raise NotImplementedError("CondPoisson is not implemented for this backend")
