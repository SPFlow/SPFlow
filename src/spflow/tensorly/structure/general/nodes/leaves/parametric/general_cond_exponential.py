from typing import List, Optional, Callable

import tensorly as tl

from spflow.meta.data import FeatureContext

from spflow.meta.data.scope import Scope


class CondExponential:  # ToDo: backend über tl.getBackend() abfragen
    def __new__(cls, scope: Scope, cond_f: Optional[Callable] = None):
        from spflow.tensorly.structure.general.nodes.leaves import CondExponential as TensorlyCondExponential
        from spflow.torch.structure.general.nodes.leaves import CondExponential as TorchCondExponential
        """TODO"""
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential(scope=scope, cond_f=cond_f)
        elif backend == "pytorch":
            return TorchCondExponential(scope=scope, cond_f=cond_f)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def accepts(cls, signatures: List[FeatureContext]) -> bool:
        from spflow.tensorly.structure.general.nodes.leaves import CondExponential as TensorlyCondExponential
        from spflow.torch.structure.general.nodes.leaves import CondExponential as TorchCondExponential
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential.accepts(signatures)
        elif backend == "pytorch":
            return TorchCondExponential.accepts(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")

    @classmethod
    def from_signatures(cls, signatures: List[FeatureContext]):
        from spflow.tensorly.structure.general.nodes.leaves import CondExponential as TensorlyCondExponential
        from spflow.torch.structure.general.nodes.leaves import CondExponential as TorchCondExponential
        backend = tl.get_backend()
        if backend == "numpy":
            return TensorlyCondExponential.from_signatures(signatures)
        elif backend == "pytorch":
            return TorchCondExponential.from_signatures(signatures)
        else:
            raise NotImplementedError("GeneralGaussian is not implemented for this backend")
